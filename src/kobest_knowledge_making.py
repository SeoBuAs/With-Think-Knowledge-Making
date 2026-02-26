from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from postprocess_output import strip_think_tags
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "kobest_knowledge_test.jsonl"
DEFAULT_MODEL = "LGAI-EXAONE/EXAONE-4.0-32B"

REPO_ID = "skt/kobest_v1"

TASKS = ("copa", "wic", "hellaswag", "sentineg", "boolq")

KNOWLEDGE_INSTRUCTION = """[System Role]
주어진 한국어 NLU 문제와 정답을 보고, 4단계 추론과 한 문장 rationale을 생성하세요.

[Output Format]
1. [Think] — 4단계 추론 (한국어, 논리적 연결):
   - Step 1: 핵심 질문·문맥을 파악하고 무엇을 묻는지 정리한다.
   - Step 2: 선택지(또는 가능한 답)를 각각 분석하고 틀린 것을 제거한다.
   - Step 3: 남은 후보를 비교하여 정답을 골라 근거를 적는다.
   - Step 4: 정답을 확정하고 오류 가능성을 배제한다.

2. [Response] — 한 문장 rationale (한국어) + 최종 답:
   - 정답의 핵심 이유를 한 문장으로 압축한다.
   - 요청된 형식(문장, 예/아니오, 선택지 문장 등)으로만 끝낸다.
   - 불필요한 수식은 쓰지 않는다."""


# -------- COPA --------


def copa_doc_to_text(doc: dict) -> str:
    connector = {"원인": " 왜냐하면", "결과": " 그래서"}.get(
        (doc.get("question") or "").strip(), " 그래서"
    )
    premise = (doc.get("premise") or "").strip()
    return f"{premise}{connector}"


def copa_doc_to_target(doc: dict) -> str:
    label = doc.get("label", 0)
    correct = doc["alternative_2"] if label == 1 else doc["alternative_1"]
    return (correct or "").strip()


def copa_doc_to_choice(doc: dict) -> list:
    return [
        (doc.get("alternative_1") or "").strip(),
        (doc.get("alternative_2") or "").strip(),
    ]


# -------- SenteNeg --------


def sentineg_doc_to_text(doc: dict) -> str:
    sentence = (doc.get("sentence") or "").strip()
    return f"문장: {sentence} 긍부정:"


def sentineg_doc_to_target(doc: dict) -> str:
    return "긍정" if doc.get("label", 0) == 1 else "부정"


# -------- WiC --------


def wic_doc_to_text(doc: dict) -> str:
    word = doc.get("word") or doc.get("target_word") or ""
    c1 = (doc.get("context_1") or "").strip()
    c2 = (doc.get("context_2") or "").strip()
    return f"문장1: {c1} 문장2: {c2} 두 문장에서 {word}가 같은 뜻으로 쓰였나?"


def wic_doc_to_target(doc: dict) -> str:
    return "예" if doc.get("label", 0) == 1 else "아니오"


# -------- HellaSwag --------


def hellaswag_process_doc(dataset: Dataset) -> Dataset:
    def preprocessor(row):
        return {
            "query": f"문장: {(row.get('context') or '').strip()}",
            "choices": [
                (row.get("ending_1") or "").strip(),
                (row.get("ending_2") or "").strip(),
                (row.get("ending_3") or "").strip(),
                (row.get("ending_4") or "").strip(),
            ],
            "gold": int(row.get("label", 0)),
        }

    return dataset.map(preprocessor, batched=False)


def hellaswag_doc_to_text(doc: dict) -> str:
    return (doc.get("query") or "").strip()


def hellaswag_doc_to_target(doc: dict) -> str:
    choices = doc.get("choices") or []
    gold = int(doc.get("gold", 0))
    if 0 <= gold < len(choices):
        return choices[gold]
    return ""


# -------- BoolQ --------


def boolq_doc_to_text(doc: dict) -> str:
    paragraph = (doc.get("paragraph") or "").strip()
    question = (doc.get("question") or "").strip()
    return f"문단: {paragraph}\n질문: {question}\n답:"


def boolq_doc_to_target(doc: dict) -> str:
    return "예" if doc.get("label", 0) == 1 else "아니오"


# -------- Task registry: doc_to_text, doc_to_target --------


def _get_task_fns(task: str):
    if task == "copa":
        return copa_doc_to_text, copa_doc_to_target
    if task == "sentineg":
        return sentineg_doc_to_text, sentineg_doc_to_target
    if task == "wic":
        return wic_doc_to_text, wic_doc_to_target
    if task == "hellaswag":
        return hellaswag_doc_to_text, hellaswag_doc_to_target
    if task == "boolq":
        return boolq_doc_to_text, boolq_doc_to_target
    raise ValueError(f"Unknown task: {task}")


def build_problem(doc: dict, task: str) -> str:
    to_text, _ = _get_task_fns(task)
    return to_text(doc)


def build_problem_text(doc: dict, task: str) -> str:
    to_text, to_target = _get_task_fns(task)
    text = to_text(doc)
    target = to_target(doc)
    return (text.rstrip() + " " + target) if target else text


def prompt_with_chat_template(problem_text: str, tokenizer) -> str:
    user_content = f"{KNOWLEDGE_INSTRUCTION}\n\n[Input]\n{problem_text}"
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# -------- Evaluation: macro F1 --------


def macro_f1_score(items):
    """items: list of (gold, pred). Returns macro F1."""
    from sklearn.metrics import f1_score

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    return f1_score(golds, preds, average="macro")


# -------- main --------


def main() -> int:
    p = argparse.ArgumentParser(
        description="KoBEST [Think] 추론 + [Response] rationale + 정답"
    )
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="test")
    p.add_argument(
        "--task",
        type=str,
        default="copa",
        choices=list(TASKS),
        help="copa, wic, hellaswag, sentineg, boolq",
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

    kwargs = {"name": args.task, "split": args.split}
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir

    print(f"[로드] {REPO_ID} task={args.task} split={args.split} ...")
    ds = load_dataset(REPO_ID, **kwargs)
    if hasattr(ds, "keys") and args.split in ds.keys():
        ds = ds[args.split]

    # HellaSwag: map to query/choices/gold
    if args.task == "hellaswag":
        ds = hellaswag_process_doc(ds)

    rows = []
    for i in range(len(ds)):
        item = ds[i]
        problem = build_problem(item, args.task)
        if not problem.strip():
            continue
        rows.append({
            "id": i,
            "problem": problem,
            "problem_text": build_problem_text(item, args.task),
        })

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
