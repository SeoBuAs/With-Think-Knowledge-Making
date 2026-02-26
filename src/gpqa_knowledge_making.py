from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

from postprocess_output import strip_think_tags
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "gpqa_knowledge_test.jsonl"
DEFAULT_MODEL = "LGAI-EXAONE/EXAONE-4.0-32B"

REPO_ID = "Idavidrein/gpqa"

CONFIGS = ("gpqa_main", "gpqa_diamond", "gpqa_extended", "gpqa_experts")

KNOWLEDGE_INSTRUCTION = """[System Role]
Given a graduate-level multiple-choice question, produce a 4-step reasoning trace and a concise rationale. Reason from domain knowledge and logic only. Do not use phrases that refer to "the passage", "the text gives", "as stated in the question", or any wording that implies the answer is already given in the stem—only derive the answer by reasoning.

[Output Format]
1. [Think] — 4-step reasoning in English (logical chain):
   - Step 1: Identify the domain and what is being asked.
   - Step 2: Analyze each option and eliminate incorrect ones.
   - Step 3: Compare remaining options and identify the correct one.
   - Step 4: Confirm the answer and rule out errors.

2. [Response] — One sentence rationale in English + final answer:
   - Rationale: Core reason for the correct answer in one sentence.
   - End with: Answer: (A) or (B) or (C) or (D)
   - Use compressed style, no filler phrases.

[Example]
Question: What is the primary mechanism by which X leads to Y in standard model Z?
A. Mechanism alpha  B. Mechanism beta  C. Mechanism gamma  D. Mechanism delta

[Think]
Step 1: Domain is X/Y/Z; we are asked for the primary mechanism.
Step 2: Alpha implies P, which contradicts known Z; beta requires condition not stated; delta is a consequence, not primary. Eliminate A, B, D.
Step 3: Gamma is the established primary pathway in Z.
Step 4: Gamma is correct; no confound.
[Response]
The primary mechanism in Z is gamma, as it directly links X to Y. Answer: (C)."""


# -------- Preprocess & process_docs --------


def preprocess(text: str | None) -> str:
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: Dataset, seed: int | None = None) -> Dataset:
    """4지선다 셔플 후 choice1~4, answer (A)~(D) 추가. question 보존."""

    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    def _process_doc(doc):
        correct_processed = preprocess(
            doc.get("Correct Answer") or doc.get("correct_answer")
        )
        choices = [
            preprocess(doc.get("Incorrect Answer 1") or doc.get("incorrect_answer_1")),
            preprocess(doc.get("Incorrect Answer 2") or doc.get("incorrect_answer_2")),
            preprocess(doc.get("Incorrect Answer 3") or doc.get("incorrect_answer_3")),
            correct_processed,
        ]
        rng.shuffle(choices)
        correct_answer_index = choices.index(correct_processed)
        question = (
            doc.get("Question")
            or doc.get("question")
            or doc.get("question_text")
            or ""
        )
        return {
            "question": question.strip(),
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": f"({chr(65 + correct_answer_index)})",
        }

    return dataset.map(_process_doc, batched=False)


# -------- doc_to_text / doc_to_target (processed doc) --------


def doc_to_text(doc: dict) -> str:
    """Question + A. choice1 ... D. choice4 + Answer:"""
    q = (doc.get("question") or "").strip()
    c1 = (doc.get("choice1") or "").strip()
    c2 = (doc.get("choice2") or "").strip()
    c3 = (doc.get("choice3") or "").strip()
    c4 = (doc.get("choice4") or "").strip()
    return f"{q}\nA. {c1}\nB. {c2}\nC. {c3}\nD. {c4}\nAnswer:"


def doc_to_target(doc: dict) -> str:
    return (doc.get("answer") or "(A)").strip()


def build_problem(doc: dict) -> str:
    return doc_to_text(doc)


def build_problem_text(doc: dict) -> str:
    return doc_to_text(doc).rstrip() + " " + doc_to_target(doc)


def prompt_with_chat_template(problem_text: str, tokenizer) -> str:
    user_content = f"{KNOWLEDGE_INSTRUCTION}\n\n[Input]\n{problem_text}"
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# -------- main --------


def main() -> int:
    p = argparse.ArgumentParser(
        description="GPQA [Think] reasoning + [Response] rationale + Answer: (A)~(D)"
    )
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42, help="process_docs 셔플 시드")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=str, default="auto")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save_id", action="store_true")
    p.add_argument("--gpu", type=str, default="2")
    p.add_argument("--save_raw", action="store_true")
    args = p.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    configs_to_run = list(CONFIGS)

    rows = []
    for cfg in configs_to_run:
        load_kw = {"name": cfg}
        if args.cache_dir:
            load_kw["cache_dir"] = args.cache_dir
        print(f"[로드] {REPO_ID} config={cfg} ...")
        ds = load_dataset(REPO_ID, **load_kw)
        if hasattr(ds, "keys"):
            split_name = list(ds.keys())[0]
            ds = ds[split_name]
        ds = process_docs(ds, seed=args.seed)
        for i in range(len(ds)):
            item = ds[i]
            problem = build_problem(item)
            if not (item.get("question") or "").strip():
                continue
            rows.append({
                "id": len(rows),
                "problem": problem,
                "problem_text": build_problem_text(item),
            })
        print(f"      config={cfg} 행 수: {len(ds)} → 누적 {len(rows)}")

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

    batch_size = n if args.batch_size.lower() == "auto" else max(1, int(args.batch_size))
    if args.batch_size.lower() == "auto":
        print(f"[배치] auto → {batch_size}")
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
