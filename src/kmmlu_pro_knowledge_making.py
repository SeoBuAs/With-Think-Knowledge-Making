from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from postprocess_output import strip_think_tags
from datasets import load_dataset
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "results" / "kmmlu_pro_knowledge.jsonl"
DEFAULT_MODEL = "LGAI-EXAONE/EXAONE-4.0-32B"

REPO_ID = "LGAI-EXAONE/KMMLU-Pro"

LABELS = ("A", "B", "C", "D", "E")

KNOWLEDGE_INSTRUCTION = """[System Role]
Given a multiple-choice question and its correct answer, produce a reasoning trace and a concise rationale.

[Output Format]
1. [Think] — 4-step reasoning in English (logical chain):
   - Step 1: Identify the key question and what is being asked.
   - Step 2: Analyze each option and eliminate wrong ones.
   - Step 3: Compare remaining options and identify the correct one.
   - Step 4: Confirm the answer and rule out errors.

2. [Response] — One sentence rationale in Korean + final answer:
   - Rationale: 핵심 정답 이유를 1~2 문장으로.
   - End with: 정답: (A 또는 B 또는 C 또는 D 또는 E)
   - Use compressed style, no filler phrases.

[Example]
[Think]
1. Key question: Which land use regulation uses performance criteria?
2. Options: PUD, Incentive Zoning, Performance Zoning, TDR. Eliminate non-performance types.
3. Performance Zoning fits: activities allowed if they meet preset criteria.
4. Confirm: Performance Zoning. Answer: C.

[Response]
성과 기준에 부합하는 활동만 허용하는 규제 방식입니다. 정답: C"""


def doc_to_text(doc: dict) -> str:
    """KMMLU/Redux 동일 Q 포맷: instruction + 문제 + A) B) C) D) [E)]"""
    question = (doc.get("question") or "").strip()
    options = doc.get("options") or []
    n = len(options)
    
    if n == 5:
        letters = "ABCDE"
    else:
        letters = "ABCD"
        
    instruction = (
        f"다음 문제에 대해 정답을 고르세요. 당신의 최종 정답은 {letters} 중 하나이고, "
        '"정답:" 뒤에 와야 합니다. 정답을 고르기 전에 차근차근 생각하고 추론하세요.\n\n'
    )
    
    option_str = ""
    for i, opt in enumerate(options):
        option_str += f"{chr(65+i)}) {opt}\n"
    option_str = option_str.strip()
    
    return instruction + f"문제:\n{question}\n\n{option_str}\n\n"


def doc_to_target(doc: dict) -> str:
    """골드 정답 글자 A~E. solution은 1-indexed (1=A, 2=B, ...)."""
    sol = doc.get("solution", 1)
    try:
        idx = int(sol) - 1
        return LABELS[max(0, min(idx, 4))]
    except (TypeError, ValueError):
        return "A"


def build_problem(doc: dict) -> str:
    """문제만 (정답 제외)."""
    return doc_to_text(doc)


def build_problem_text(doc: dict) -> str:
    """문제 + 정답 (LLM 입력용)."""
    return doc_to_text(doc) + "정답: " + doc_to_target(doc)


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
        description="KMMLU-Pro [Think] 4-step English + [Response] Korean rationale + 정답: A~E"
    )
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="test", help="test (KMMLU-Pro는 test만 제공)")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=str, default="auto")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save_id", action="store_true")
    p.add_argument("--gpu", type=str, default="3")
    p.add_argument("--save_raw", action="store_true")
    args = p.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    kwargs = {"split": args.split}
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir
    # KMMLU-Pro는 config name이 있을 수 있음
    try:
        ds = load_dataset(REPO_ID, **kwargs)
    except Exception:
        ds = load_dataset(REPO_ID, name="kmmlu_pro", **kwargs)

    rows = []
    for i in range(len(ds)):
        item = ds[i]
        q = (item.get("question") or "").strip()
        opts = item.get("options") or []
        if not q or len(opts) < 4:
            continue
        rows.append({
            "id": i,
            "problem": build_problem(item),
            "problem_text": build_problem_text(item),
        })

    if args.limit is not None:
        rows = rows[: args.limit]
    n = len(rows)
    print(f"      행 수: {n}")

    if n == 0:
        print("처리할 행이 없습니다.")
        return 0

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
