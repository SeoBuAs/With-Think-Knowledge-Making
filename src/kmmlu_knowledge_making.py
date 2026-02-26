from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from postprocess_output import strip_think_tags
from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "results" / "kmmlu_knowledge.jsonl"
DEFAULT_MODEL = "LGAI-EXAONE/EXAONE-4.0-32B"

REPO_ID = "HAERAE-HUB/KMMLU"

LABELS = ("A", "B", "C", "D")

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
   - End with: 정답: (A 또는 B 또는 C 또는 D)
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
    """KMMLU 포맷: question + A. {{A}}\\nB. {{B}}\\nC. {{C}}\\nD. {{D}}\\n정답："""
    q = (doc.get("question") or "").strip()
    a = (doc.get("A") or "").strip()
    b = (doc.get("B") or "").strip()
    c = (doc.get("C") or "").strip()
    d = (doc.get("D") or "").strip()
    return f"{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n정답："


def doc_to_target(doc: dict) -> str:
    """정답 라벨 A/B/C/D. answer는 1=A, 2=B, 3=C, 4=D."""
    ans = doc.get("answer")
    if ans is None:
        return "A"
    try:
        idx = int(ans)
        return LABELS[max(0, min(idx - 1, 3))]
    except (TypeError, ValueError):
        s = str(ans).strip().upper()
        if s in LABELS:
            return s
        if s in ("1", "2", "3", "4"):
            return LABELS[int(s) - 1]
        return "A"


def build_problem(doc: dict) -> str:
    """문제만 (정답 제외)."""
    return doc_to_text(doc)


def build_problem_text(doc: dict) -> str:
    """문제 + 정답 (LLM 입력용)."""
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
        description="KMMLU [Think] 4-step English + [Response] Korean rationale + 정답: A/B/C/D"
    )
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="all", help="train, validation, test, or all (train+validation+test)")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="단일 config(과목)만 로드. 미지정 시 전체 config 로드",
    )
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

    kwargs_base = {}
    if args.cache_dir:
        kwargs_base["cache_dir"] = args.cache_dir

    if args.config:
        config_list = [args.config]
        print(f"[로드] {REPO_ID} config={args.config} split={args.split} ...")
    else:
        config_list = get_dataset_config_names(REPO_ID)
        print(f"[로드] {REPO_ID} config 수: {len(config_list)}, split={args.split} ...")

    if args.split.lower() == "all":
        split_names = ["train", "validation", "test"]
    else:
        split_names = [args.split]

    rows = []
    global_id = 0
    for cfg in config_list:
        for sp in split_names:
            try:
                ds = load_dataset(REPO_ID, name=cfg, split=sp, **kwargs_base)
            except (ValueError, Exception):
                continue
            if ds is None or len(ds) == 0:
                continue
            for i in range(len(ds)):
                item = ds[i]
                q = (item.get("question") or "").strip()
                if not q:
                    continue
                a, b, c, d = item.get("A"), item.get("B"), item.get("C"), item.get("D")
                if not any(x for x in (a, b, c, d) if x):
                    continue
                rows.append({
                    "id": global_id,
                    "problem": build_problem(item),
                    "problem_text": build_problem_text(item),
                })
                global_id += 1
            if not args.config:
                print(f"  {cfg} {sp}: +{len(ds)} (누적 {len(rows)})")

    if args.limit is not None:
        rows = rows[: args.limit]
    n = len(rows)
    print(f"      총 행 수: {n}")

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
