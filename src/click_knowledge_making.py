from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

from postprocess_output import strip_think_tags
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "results" / "click_cot_short.jsonl"
DEFAULT_MODEL = "LGAI-EXAONE/EXAONE-4.0-32B"

REPO_ID = "EunsuKim/CLIcK"

LABELS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")

KNOWLEDGE_INSTRUCTION = """[System Role]
Given a QA item with context, question, options, and the correct answer, produce a reasoning trace and a concise rationale.

[Output Format]
1. [Think] — 4-step reasoning in English (logical chain):
   - Step 1: Identify the key question and core concepts.
   - Step 2: Extract relevant facts, laws, or rules from the context.
   - Step 3: Connect the correct option to the evidence.
   - Step 4: Rule out wrong options or confirm the answer.

2. [Response] — One sentence rationale in Korean + final answer:
   - Rationale: 핵심 이유 또는 관련 법규/공식을 한 문장으로.
   - End with: 정답: (A or B or C or D)
   - Use compressed style, no filler phrases.

[Example]
[Think]
1. Key question: Year Korea overcame FX crisis.
2. Context: IMF bailout 1997, program completion.
3. Link: Official end = loan repayment = 2001.
4. Conclusion: C (2001).

[Response]
한국은 2001년 IMF 구제금융 조기 상환으로 외환위기 공식 종료. 정답: C"""


# --- QA 포맷 (get_context, get_target, get_choices) ---

def get_context(doc: dict) -> str:
    """맥락/질문/보기 포맷 문자열 (정답 없음)."""
    ctx = doc.get("paragraph") or ""
    q = doc.get("question") or ""
    opt = doc.get("choices") or []
    opt_fmt = ", ".join(f"{chr(65 + i)}: {opt[i]}" for i in range(min(4, len(opt))))
    if len(opt) >= 5:
        opt_fmt += f", E: {opt[4]}"
    if ctx.strip():
        return f"주어진 맥락을 천천히 읽고, 질문에 대한 적절한 정답을 A, B, C, D 중에 골라 알파벳 하나로 답하시오.\n\n맥락: {ctx}\n질문: {q}\n보기:\n{opt_fmt}\n정답:"
    return f"주어진 질문을 천천히 읽고, 적절한 정답을 A, B, C, D 중에 골라 알파벳 하나로 답하시오.\n\n질문: {q}\n보기:\n{opt_fmt}\n정답:"


def get_target(doc: dict) -> str:
    """정답을 A/B/C/D(/E) 라벨로 반환."""
    ans = doc.get("answer")
    choices = doc.get("choices") or []
    if not ans or not choices:
        return ""
    labels = ["A", "B", "C", "D", "E"] if "CSAT" in str(doc.get("id", "")) else ["A", "B", "C", "D"]
    try:
        idx = choices.index(ans)
        return labels[idx] if idx < len(labels) else str(ans)
    except (ValueError, TypeError):
        return str(ans)


def get_choices(doc: dict) -> List[str]:
    """선지 라벨 목록."""
    if "CSAT" in str(doc.get("id", "")):
        return ["A", "B", "C", "D", "E"]
    return ["A", "B", "C", "D"]


# --- 필터 함수 ---

def _extract_text(ds: Dataset) -> Dataset:
    return ds.filter(
        lambda ex: "CSAT_korean_22" in ex["id"]
        or ("CSAT_korean_23" in ex["id"] and int(ex["id"].split("_")[-1]) < 35)
        or ("TK" in ex["id"] and int(ex["id"].split("_")[-1]) > 4)
    )


def _extract_grammar(ds: Dataset) -> Dataset:
    return ds.filter(
        lambda ex: (
            "CSAT_korean" in ex["id"]
            and int(ex["id"].split("_")[2]) < 21
            and int(ex["id"].split("_")[3]) > 10
        )
        or (
            "Kedu_1" in ex["id"]
            and (
                ex["id"].split("_")[1] != "16"
                or not any(k in (ex.get("question") or "") for k in ("대화", "발화", "질의"))
            )
        )
        or ("TK" in ex["id"] and int(ex["id"].split("_")[-1]) < 5)
    )


def _extract_function(ds: Dataset) -> Dataset:
    return ds.filter(
        lambda ex: (
            "CSAT_korean" in ex["id"]
            and (
                int(ex["id"].split("_")[-1]) > 34
                or (
                    int(ex["id"].split("_")[2]) < 21
                    and int(ex["id"].split("_")[3]) < 11
                )
            )
        )
        or (
            "Kedu_16" in ex["id"]
            and any(k in (ex.get("question") or "") for k in ("대화", "발화", "질의"))
        )
        or "PSE_korean" in ex["id"]
    )


def _extract_economy(ds: Dataset) -> Dataset:
    return ds.filter(lambda ex: "economy" in str(ex["id"]).lower())


def _extract_geography(ds: Dataset) -> Dataset:
    return ds.filter(lambda ex: "geography" in str(ex["id"]).lower())


def _extract_history(ds: Dataset) -> Dataset:
    return ds.filter(
        lambda ex: "KHB" in ex["id"] or "history" in str(ex["id"]).lower()
    )


def _extract_law(ds: Dataset) -> Dataset:
    return ds.filter(
        lambda ex: "law" in str(ex["id"]).lower() or "PSAT" in ex["id"]
    )


def _extract_politics(ds: Dataset) -> Dataset:
    return ds.filter(lambda ex: "politics" in str(ex["id"]).lower())


def _extract_kpop(ds: Dataset) -> Dataset:
    return ds.filter(lambda ex: "popular" in str(ex["id"]).lower())


def _extract_society(ds: Dataset) -> Dataset:
    return ds.filter(lambda ex: "society" in str(ex["id"]).lower())


def _extract_tradition(ds: Dataset) -> Dataset:
    return ds.filter(lambda ex: "tradition" in str(ex["id"]).lower())


EXTRACTORS = {
    "text": _extract_text,
    "grammar": _extract_grammar,
    "function": _extract_function,
    "economy": _extract_economy,
    "geography": _extract_geography,
    "history": _extract_history,
    "law": _extract_law,
    "politics": _extract_politics,
    "kpop": _extract_kpop,
    "society": _extract_society,
    "tradition": _extract_tradition,
}


def build_problem(doc: dict) -> str:
    """문제만 (정답 제외, inference용)."""
    return get_context(doc)


def build_problem_text(doc: dict) -> str:
    """문제 + 정답 (LLM 입력용). get_context + 정답 라벨."""
    ctx = get_context(doc)
    tgt = get_target(doc)
    return ctx.rstrip() + (" " + tgt if tgt else "")


def prompt_with_chat_template(problem_text: str, tokenizer) -> str:
    """지시문 + [Input] 문제 텍스트. think 태그 미사용."""
    user_content = f"{KNOWLEDGE_INSTRUCTION}\n\n[Input]\n{problem_text}"
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="CLIcK [Think] 4-step English + [Response] 한국어 rationale+정답 생성"
    )
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument(
        "--batch_size",
        type=str,
        default="auto",
        help="배치 크기. 정수 또는 'auto'",
    )
    p.add_argument(
        "--filter",
        type=str,
        default="all",
        choices=["all"] + list(EXTRACTORS.keys()),
        help="서브셋 필터. all=필터 없음",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save_id", action="store_true", help="저장 시 id 필드 포함")
    p.add_argument("--gpu", type=str, default="3", help="사용할 GPU 번호")
    p.add_argument(
        "--save_raw",
        action="store_true",
        help="중간 단계 raw 파일 저장 (prompts_raw, outputs_raw)",
    )
    args = p.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"[로드] {REPO_ID} split={args.split} ...")
    kwargs = {"split": args.split}
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir
    ds = load_dataset(REPO_ID, **kwargs)

    if args.filter != "all":
        ds = EXTRACTORS[args.filter](ds)
        print(f"      필터 적용: {args.filter} → {len(ds)} 행")

    rows = []
    for i in range(len(ds)):
        item = ds[i]
        if not (item.get("question") or item.get("answer")):
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
        temperature=0.1,
        top_p=0.5,
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
