# With-Think Knowledge Making

벤치마크별로 **문제 + 정답**을 입력해, 강한 모델(기본: EXAONE-4.0-32B)이 **[Think]** 4단계 추론과 **[Response]** 요약 정답을 생성하도록 하는 스크립트 모음입니다. 생성된 JSONL은 학습/캘리브레이션/Distillation용 “knowledge” 데이터로 사용할 수 있습니다.

---

## 출력 형식

모든 스크립트는 동일한 형식을 요구합니다.

- **[Think]** — 4-step reasoning (영어, 논리 체인)
- **[Response]** — 한 문장 요약 + 최종 정답 (과제에 따라 영어/한국어, `The answer is N` 또는 `정답: A` 등)

생성 후 `postprocess_output.strip_think_tags`로 `[Think]`/`[Response]` 구간을 정리해 저장합니다.

---

## 스크립트 목록

| 스크립트 | 데이터셋 | 출력 예시 경로 |
|----------|----------|----------------|
| `gsm8k_knowledge_making.py` | openai/gsm8k | `../results/gsm8k_knowledge.jsonl` |
| `gpqa_knowledge_making.py` | Idavidrein/gpqa | `results/gpqa_knowledge_test.jsonl` |
| `hrm8k_knowledge_making.py` | HRM8K | `../results/hrm8k_knowledge.jsonl` |
| `click_knowledge_making.py` | EunsuKim/CLIcK | `../results/click_cot_short.jsonl` |
| `kmmlu_knowledge_making.py` | KMMLU | `../results/kmmlu_knowledge.jsonl` |
| `kmmlu_pro_knowledge_making.py` | LGAI-EXAONE/KMMLU-Pro | `../results/kmmlu_pro_knowledge.jsonl` |
| `kmmlu_redux_knowledge_making.py` | LGAI-EXAONE/KMMLU-Redux | `../results/kmmlu_redux_knowledge.jsonl` |
| `kobest_knowledge_making.py` | KOBEST | `../results/kobest_knowledge.jsonl` |
| `mmlu_knowledge_making.py` | MMLU | `../results/mmlu_knowledge.jsonl` |
| `mmlu_redux_knowledge_making.py` | MMLU-Redux | `../results/mmlu_redux_knowledge.jsonl` |

각 스크립트 내부의 `DEFAULT_OUTPUT`이 실제 기본 출력 경로입니다.

---

## 디렉터리 구조

```
with_think_src/
├── README.md
├── gsm8k_knowledge_making.py
├── gpqa_knowledge_making.py
├── hrm8k_knowledge_making.py
├── click_knowledge_making.py
├── kmmlu_knowledge_making.py
├── kmmlu_pro_knowledge_making.py
├── kmmlu_redux_knowledge_making.py
├── kobest_knowledge_making.py
├── mmlu_knowledge_making.py
├── mmlu_redux_knowledge_making.py
└── (postprocess_output 모듈은 PYTHONPATH 또는 동일 디렉터리에 필요)
```

---

## 사용 방법

**실행 위치**: 스크립트가 `postprocess_output`를 import하므로, 해당 모듈이 있는 디렉터리에서 실행하거나 `PYTHONPATH`에 포함해야 합니다.

```bash
cd /path/to/C_data_creation/with_think_src   # 또는 postprocess_output이 보이는 경로

# GSM8K (기본: train+validation, 출력은 ../results/gsm8k_knowledge.jsonl)
python gsm8k_knowledge_making.py --gpu 0

# KMMLU-Pro (일부만 생성 시 --limit)
python kmmlu_pro_knowledge_making.py --gpu 0 --output ../results/kmmlu_pro_knowledge.jsonl --limit 100

# GPQA (gated 가능성 있음, --split 등 옵션은 각 스크립트 --help 참고)
python gpqa_knowledge_making.py --gpu 0
```

공통 옵션 예시 (스크립트마다 다름):

- `--output` : 출력 JSONL 경로  
- `--model` : 사용 모델 (기본 `LGAI-EXAONE/EXAONE-4.0-32B`)  
- `--gpu` : GPU 번호  
- `--limit` : 생성할 샘플 수 제한  
- `--split` : 데이터셋 스플릿 (train / validation / test / all 등)

자세한 옵션은 `python <스크립트> --help` 로 확인하세요.

---

## 의존성

- Python 3
- `datasets`, `transformers`
- vLLM (스크립트 내 생성용으로 사용 시)
- `postprocess_output` 모듈 (같은 레포 또는 패키지에 `strip_think_tags` 제공)

---

## 참고

- 각 스크립트는 벤치마크별 `doc_to_text` / `doc_to_target` 형식에 맞춰 문제·정답을 구성하고, 위 **[Think]** / **[Response]** 형식으로 생성합니다.
- gated 데이터셋(GPQA 등)은 Hugging Face 토큰이 필요할 수 있습니다.
