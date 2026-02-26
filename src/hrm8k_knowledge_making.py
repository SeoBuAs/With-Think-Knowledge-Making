from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from postprocess_output import strip_think_tags
from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from datasets.features import Value
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "results" / "hrm8k_knowledge.jsonl"
DEFAULT_MODEL = "LGAI-EXAONE/EXAONE-4.0-32B"

REPO_ID = "HAERAE-HUB/HRM8K"

KNOWLEDGE_INSTRUCTION = """[System Role]
Given a math problem and its correct answer, produce a reasoning trace and a concise rationale.

[Output Format]
1. [Think] — 4-step reasoning in English (logical chain):
   - Step 1: Identify the key question and given quantities.
   - Step 2: Decide which formulas or solution steps apply.
   - Step 3: Execute calculations or derive the result.
   - Step 4: Confirm the answer and rule out errors.

2. [Response] — One sentence rationale in Korean + final answer:
   - Rationale: 핵심 풀이 이유를 한 문장으로.
   - End with: 정답: [수치] or $\\boxed{N}$
   - Use compressed style, no filler phrases.

[Example]
[Think]
1. Key question: Find area of rectangle; length=2×width, diagonal=5√5.
2. Use Pythagorean theorem: d² = L² + W², L=2W → d² = 5W².
3. (5√5)² = 125 = 5W² → W=5, L=10. Area = 50.
4. Confirm: √(10²+5²) = √125 = 5√5. Answer: 50.

[Response]
직사각형 대각선 d=√(L²+W²), L=2W이므로 5√5=W√5 → W=5, L=10. 넓이 50. 정답: 50"""


# --- doc_to_text, doc_to_target, postprocess ---

def doc_to_text(doc: dict) -> str:
    """문제 포맷 (정답 없음)."""
    q = (doc.get("question") or "").strip()
    return (
        "주어진 문제를 풀어보세요.\n"
        "문제를 푼 후, 최종 답변을 다음과 같은 형식으로 작성하세요: $\\boxed{N}$.\n\n"
        f"문제: {q}\n답변:"
    )


def postprocess(s) -> str:
    s = str(s).strip()
    try:
        float_value = float(s)
        return str(int(float_value)) if float_value == int(float_value) else str(float_value)
    except Exception:
        return s


def doc_to_target(doc: dict) -> str:
    return postprocess(doc.get("answer", ""))


# --- Math parsing (parse_math_answer, is_equiv, _strip_string 등) ---

def parse_math_answer(raw_string: str):
    if not raw_string or not str(raw_string).strip():
        return None

    def remove_boxed(s):
        left = "\\boxed{"
        try:
            if s[: len(left)] != left or s[-1] != "}":
                return None
            answer = s[len(left) : -1]
            if "=" in answer:
                answer = answer.split("=")[-1].lstrip(" ")
            return answer.strip()
        except Exception:
            return None

    def last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None
        i = idx
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    return string[idx : i + 1]
            i += 1
        return None

    def get_answer_with_dollar_sign(s):
        matches = re.findall(r"\$(.*)\$", s)
        if matches:
            last_match = matches[-1].strip()
            if "=" in last_match:
                last_match = last_match.split("=")[-1].lstrip(" ")
            return last_match
        return None

    def get_answer_without_dollar_sign(s):
        if "=" in s:
            last_match = s.split("=")[-1].lstrip(" ").rstrip(".")
            if "\\n" in last_match:
                last_match = last_match.split("\\n")[0]
            return last_match.strip() if last_match else None
        matches = re.findall(r"(?:\$)?\d+(?:\.\d+)?(?![\\w\\d])", s)
        return matches[-1] if matches else None

    s = str(raw_string)
    if "\\boxed" in s:
        boxed = last_boxed_only_string(s)
        if boxed:
            return remove_boxed(boxed)
    ans = get_answer_with_dollar_sign(s)
    if not ans:
        ans = get_answer_without_dollar_sign(s)
    return ans


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            elif len(substr) >= 2:
                a, b = substr[0], substr[1]
                post = substr[2:] if len(substr) > 2 else ""
                new_str += ("{" + a + "}{" + b + "}" + post) if b != "{" else ("{" + a + "}" + b + post)
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")[0], string.split("/")[1]
    try:
        ai, bi = int(a), int(b)
        if string == f"{ai}/{bi}" or string == f"{a}/{b}":
            return f"\\frac{{{ai}}}{{{bi}}}"
    except Exception:
        pass
    return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def is_equiv(str1, str2, verbose=False) -> bool:
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    s1 = parse_math_answer(str1) if ("\\boxed" in str(str1) or "\\frac" in str(str1)) else str(str1).strip()
    s2 = parse_math_answer(str2) if ("\\boxed" in str(str2) or "\\frac" in str(str2)) else str(str2).strip()
    if s1 is None:
        s1 = str(str1).strip()
    if s2 is None:
        s2 = str(str2).strip()
    try:
        f1, f2 = float(s1), float(s2)
        if abs(f1 - f2) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    try:
        ss1 = postprocess(_strip_string(str(s1)))
        ss2 = postprocess(_strip_string(str(s2)))
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str(s1) == str(s2)


# --- build_problem, build_problem_text ---

def build_problem(doc: dict) -> str:
    """문제만 (정답 제외)."""
    return doc_to_text(doc)


def build_problem_text(doc: dict) -> str:
    """문제 + 정답 (LLM 입력용)."""
    return doc_to_text(doc) + " " + doc_to_target(doc)


def prompt_with_chat_template(problem_text: str, tokenizer) -> str:
    """지시문 + [Input]."""
    user_content = f"{KNOWLEDGE_INSTRUCTION}\n\n[Input]\n{problem_text}"
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="HRM8K [Think] 4-step English + [Response] 한국어 rationale+정답 생성"
    )
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--name", type=str, default=None, help="서브셋 (GSM8K, KSM, MATH 등). 미지정 시 전체")
    p.add_argument("--split", type=str, default="test", help="train, test, or all (train+test)")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=768)
    p.add_argument(
        "--batch_size",
        type=str,
        default="auto",
        help="배치 크기. 정수 또는 'auto'",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save_id", action="store_true", help="저장 시 id 필드 포함")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument(
        "--save_raw",
        action="store_true",
        help="중간 단계 raw 파일 저장",
    )
    args = p.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"[로드] {REPO_ID} split={args.split} ...")
    kwargs_base = {}
    if args.cache_dir:
        kwargs_base["cache_dir"] = args.cache_dir

    rows = []
    if args.split.lower() == "all":
        for sp in ("train", "test"):
            if args.name:
                ds = load_dataset(REPO_ID, name=args.name, split=sp, **kwargs_base)
            else:
                parts = []
                for cfg in get_dataset_config_names(REPO_ID):
                    try:
                        sub = load_dataset(REPO_ID, name=cfg, split=sp, **kwargs_base)
                        if sub.num_rows > 0:
                            for col in ("answer", "difficulty"):
                                if col in sub.column_names:
                                    sub = sub.cast_column(col, Value("string"))
                            parts.append(sub)
                    except Exception:
                        continue
                ds = concatenate_datasets(parts) if parts else None
            if ds is None or len(ds) == 0:
                continue
            for i in range(len(ds)):
                item = ds[i]
                q = (item.get("question") or "").strip()
                ans = item.get("answer")
                if not q or ans is None:
                    continue
                rows.append({
                    "id": item.get("id", f"{sp}_{i}"),
                    "problem": build_problem(item),
                    "problem_text": build_problem_text(item),
                })
    else:
        if args.name:
            ds = load_dataset(REPO_ID, name=args.name, split=args.split, **kwargs_base)
        else:
            parts = []
            for cfg in get_dataset_config_names(REPO_ID):
                try:
                    sub = load_dataset(REPO_ID, name=cfg, split=args.split, **kwargs_base)
                    if sub.num_rows > 0:
                        for col in ("answer", "difficulty"):
                            if col in sub.column_names:
                                sub = sub.cast_column(col, Value("string"))
                        parts.append(sub)
                except Exception:
                    continue
            ds = concatenate_datasets(parts) if parts else None
        if ds is None or len(ds) == 0:
            raise SystemExit(f"{REPO_ID} split={args.split} 로드 실패 (유효한 행 없음)")
        for i in range(len(ds)):
            item = ds[i]
            q = (item.get("question") or "").strip()
            ans = item.get("answer")
            if not q or ans is None:
                continue
            rows.append({
                "id": item.get("id", i),
                "problem": build_problem(item),
                "problem_text": build_problem_text(item),
            })

    if not rows:
        raise SystemExit(f"{REPO_ID} 로드 실패 (유효한 행 없음)")
    if args.limit is not None:
        rows = rows[: args.limit]
    n = len(rows)
    print(f"      행 수: {n}")

    tokenizer_path = args.tokenizer or args.model
    print(f"[토크나이저] {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, cache_dir=args.cache_dir or None
    )

    print(f"[vLLM] 모델 로드: {args.model}")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        max_tokens=args.max_tokens,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.stem
    raw_dir = out_path.parent
    prompts_raw_path = raw_dir / f"{stem}_prompts_raw.jsonl" if args.save_raw else None
    outputs_raw_path = raw_dir / f"{stem}_outputs_raw.jsonl" if args.save_raw else None
    if args.save_raw:
        print(f"[raw] prompts → {prompts_raw_path.name}, outputs → {outputs_raw_path.name}")

    if args.batch_size.lower() == "auto":
        batch_size = n
        print(f"[배치] auto → {batch_size}")
    else:
        batch_size = max(1, int(args.batch_size))
    written = 0

    with open(out_path, "w", encoding="utf-8") as f:
        f_prompts_raw = open(prompts_raw_path, "w", encoding="utf-8") if prompts_raw_path else None
        f_outputs_raw = open(outputs_raw_path, "w", encoding="utf-8") if outputs_raw_path else None
        try:
            for start in range(0, n, batch_size):
                batch = rows[start : start + batch_size]
                prompts = [prompt_with_chat_template(r["problem_text"], tokenizer) for r in batch]
                if f_prompts_raw:
                    for idx, (row, prompt) in enumerate(zip(batch, prompts)):
                        rec = {"idx": start + idx, "prompt": prompt, "problem_text": row["problem_text"]}
                        if args.save_id and row.get("id") is not None:
                            rec["id"] = row["id"]
                        f_prompts_raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"  생성 중 {start + 1}..{start + len(batch)} / {n}")
                outputs = llm.generate(prompts, sampling_params)
                for idx, (row, out) in enumerate(zip(batch, outputs)):
                    raw_text = (out.outputs[0].text if out.outputs else "")
                    text = strip_think_tags(raw_text.strip())
                    save = {"problem": row["problem"], "knowledge": text}
                    if args.save_id and row.get("id") is not None:
                        save["id"] = row["id"]
                    f.write(json.dumps(save, ensure_ascii=False) + "\n")
                    written += 1
                    if f_outputs_raw:
                        rec = {"idx": start + idx, "raw_output": raw_text, "problem": row["problem"]}
                        if args.save_id and row.get("id") is not None:
                            rec["id"] = row["id"]
                        f_outputs_raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
        finally:
            if f_prompts_raw:
                f_prompts_raw.close()
            if f_outputs_raw:
                f_outputs_raw.close()
        f.flush()

    print(f"저장: {out_path.resolve()} ({written} rows)")
    if args.save_raw and prompts_raw_path and outputs_raw_path:
        print(f"  raw: {prompts_raw_path.resolve()}, {outputs_raw_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
