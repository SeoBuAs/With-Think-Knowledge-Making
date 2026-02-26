from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from postprocess_output import strip_think_tags
from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "results" / "gsm8k_knowledge.jsonl"
DEFAULT_MODEL = "LGAI-EXAONE/EXAONE-4.0-32B"

REPO_ID = "openai/gsm8k"
DATASET_CONFIG = "main"

KNOWLEDGE_INSTRUCTION = """[System Role]
Given a math word problem and its correct answer, produce a reasoning trace and a concise rationale.

[Output Format]
1. [Think] — 4-step reasoning in English (logical chain):
   - Step 1: Identify the key question and given quantities.
   - Step 2: Decide which operations or solution steps apply.
   - Step 3: Execute calculations step by step.
   - Step 4: Confirm the answer and rule out errors.

2. [Response] — One sentence rationale + final answer:
   - Rationale: Core reasoning in one sentence (English only).
   - End with: The answer is [number].
   - Use compressed style, no filler phrases.

[Example]
[Think]
1. Key question: Total clips sold in April and May; April=48, May=half of April.
2. May = 48/2 = 24. Total = April + May = 48 + 24.
3. 48 + 24 = 72.
4. Confirm: 48 + 24 = 72. Answer: 72.

[Response]
May: 48/2=24 clips. Total: 48+24=72. The answer is 72."""


# --- doc_to_text, doc_to_target (gsm8k_cot 형식) ---

def doc_to_text(doc: dict) -> str:
    """lm-eval gsm8k_cot 형식: Q: {question}\\n\\n A:"""
    q = (doc.get("question") or "").strip()
    return f"Q: {q}\n\n A:"


def postprocess(s) -> str:
    s = str(s).strip()
    try:
        float_value = float(s)
        return str(int(float_value)) if float_value == int(float_value) else str(float_value)
    except Exception:
        return s


def doc_to_target(doc: dict) -> str:
    """GSM8K answer: ...\\n#### 72 → 72 추출"""
    ans = doc.get("answer") or ""
    if "####" in ans:
        return postprocess(ans.split("####")[-1].strip())
    return postprocess(ans)


# --- Math parsing ---

def parse_math_answer(raw_string: str):
    if not raw_string or not str(raw_string).strip():
        return None
    # "The answer is 72." → 72
    m = re.search(r"The answer is\s+(-?[\d.,]+)", str(raw_string), re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "").strip()
    # #### 72
    if "####" in str(raw_string):
        return postprocess(str(raw_string).split("####")[-1].strip())
    # 숫자만
    matches = re.findall(r"-?[\d.,]+", str(raw_string))
    return matches[-1].replace(",", "").strip() if matches else None


def is_equiv(str1, str2) -> bool:
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    s1 = parse_math_answer(str1) or str(str1).strip()
    s2 = parse_math_answer(str2) or str(str2).strip()
    try:
        f1, f2 = float(str(s1).replace(",", "")), float(str(s2).replace(",", ""))
        return abs(f1 - f2) < 1e-6
    except (ValueError, TypeError):
        return str(s1) == str(s2)


# --- build_problem, build_problem_text ---

def build_problem(doc: dict) -> str:
    """문제만 (정답 제외)."""
    return doc_to_text(doc)


def build_problem_text(doc: dict) -> str:
    """문제 + 정답 (LLM 입력용). Q: ... A: [answer]"""
    return doc_to_text(doc) + " " + doc_to_target(doc)


def prompt_with_chat_template(problem_text: str, tokenizer) -> str:
    user_content = f"{KNOWLEDGE_INSTRUCTION}\n\n[Input]\n{problem_text}"
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="GSM8K [Think] 4-step English + [Response] rationale + The answer is N."
    )
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="all", help="train, validation, test, or all (train+validation)")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=str, default="auto")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save_id", action="store_true")
    p.add_argument("--gpu", type=str, default="1")
    p.add_argument("--save_raw", action="store_true")
    args = p.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"[로드] {REPO_ID} config={DATASET_CONFIG} split={args.split} ...")
    kwargs_base = {"name": DATASET_CONFIG}
    if args.cache_dir:
        kwargs_base["cache_dir"] = args.cache_dir

    if args.split.lower() == "all":
        ds_train = load_dataset(REPO_ID, split="train", **kwargs_base)
        try:
            ds_other = load_dataset(REPO_ID, split="validation", **kwargs_base)
        except ValueError:
            ds_other = load_dataset(REPO_ID, split="test", **kwargs_base)
        ds = concatenate_datasets([ds_train, ds_other])
    else:
        ds = load_dataset(REPO_ID, split=args.split, **kwargs_base)

    rows = []
    for i in range(len(ds)):
        item = ds[i]
        q = (item.get("question") or "").strip()
        ans = item.get("answer")
        if not q or not ans or "####" not in str(ans):
            continue
        rows.append({
            "id": item.get("id", i),
            "problem": build_problem(item),
            "problem_text": build_problem_text(item),
        })

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
