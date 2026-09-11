# Small-Model Math Reasoning — 5th / 42 teams, Inter-University Deep Learning Challenge 2026

Competition project: make **Qwen2.5-3B-Instruct** (fixed by the rules, no tools at inference)
solve competition-style math problems and output exact integer answers.

**Final result: 5th of 42 teams (accuracy 0.7200), up from 11th on the public leaderboard.**
Final rankings were verified by the organizers re-running every team's inference from its
public repository — this repo reproduced exactly as committed.

## The approach in one paragraph

The final model has **zero additional training**: stock Qwen2.5-3B-Instruct plus
**confidence-weighted self-consistency** (32 samples per problem at temperature 0.7, votes
weighted by the model's own token probabilities). That single axis was worth **+13.7 points
over greedy decoding** (0.648 → 0.785 on the leaderboard). Nine fine-tuning attempts
(SFT ×3, GRPO ×2, DPO, distillation ×2, full FT on 270k filtered problems) all degraded
accuracy — decomposing why revealed that training left capability intact (pass@32: 405 → 408)
while scattering answers (5.1 → 6.5 distinct answers/problem) and collapsing the majority
vote (380 → 338). A paper on this **vote-dilution** failure mode is in progress.

Key practices that made 93 experiments trustworthy on a ~$220 single-GPU budget:

- **Measured noise floor**: identical config resubmitted → 0.78459 vs 0.77978 (−0.48 pts).
  Differences under 1 pt were treated as "no difference."
- **Preregistration**: success criteria frozen in writing before each experiment ran
  (see `experiments/prereg_*.md`).
- **Data auditing**: 1,065 corrupted training items surfaced by cross-checking labels
  against the model's 32-vote consensus; all local evals ran on a corrected answer key.
- **Failure-proof final day**: code frozen a week early, resumable generation
  (checkpoint every 100 problems), a mechanical validator stress-tested on 17 pathological
  inputs before anything could be submitted.

## Reproduce the final submission (2 commands + validation)

Hardware: one 24 GB GPU (A10G used in practice), vLLM. No trained weights needed —
the model downloads from Hugging Face as-is.

```bash
# 1) sample 32 solutions per problem (resumable; ~2.5 h for 2,000 problems on an A10G)
uv run python remote/dump_lb_samples.py --n 32 --temp 0.7 --top-p 0.8 --seed 42 \
  --lb-csv deep-learning-challenge-2026/<test>.csv --out results/final_samples.jsonl

# 2) confidence-weighted vote -> submission csv
uv run python remote/make_submission_from_dump.py --dump results/final_samples.jsonl \
  --rule weighted --n 32 --lb-csv deep-learning-challenge-2026/<test>.csv --tag final

# 3) mechanical pre-submission validation (must exit 0)
uv run python remote/validate_submission.py --sub results/submission_final.csv \
  --test deep-learning-challenge-2026/<test>.csv --dump results/final_samples.jsonl --n 32
```

The exact runbook used on the final day is `docs/final-pipeline.md`.

## Map of the repo

| Path | What it is |
|---|---|
| `EXPERIMENTS.md` | **All 93 experiments, failures included** — the core log |
| `docs/final-pipeline.md` | Frozen final-day runbook (the commands above, with rationale) |
| `experiments/prereg_*.md` | Preregistration documents (success gates written before running) |
| `remote/` | Inference / training / validation scripts (vLLM-based) |
| `report.html` | Experiment report with charts and references |
| `docs/research.md` | Literature notes (Self-Consistency, GRPO, DPO, NuminaMath, …) |
| `CONTEXT.md`, `prd.md` | Project context and competition rules summary (Korean) |

---

# (한국어) 아주 소중한 딥러닝 챌린지 2026 — 수학 추론

Qwen2.5-3B-Instruct를 베이스로 수학 문제의 정수 답을 추론하는 대회 프로젝트.
**최종 5위 / 42팀 (정확도 0.7200, 리더보드 11위에서 상승)** — 순위는 운영진이 본 저장소
코드로 전체 추론을 재수행해 검증함. 규칙·평가 방식은 [prd.md](prd.md), 실험 이력은
[EXPERIMENTS.md](EXPERIMENTS.md) 참조.

## 최종 스택

**무학습 베이스 + 확신도 가중 Self-Consistency (n=32, temperature 0.7, top_p 0.8, seed 42)**

- 학습 9회 시도(SFT×3, GRPO×2, DPO, 증류×2, 27만 문항 full FT) 전부 성능 하락으로 기각.
  원인 분해: 능력(pass@32) 유지, 답 다양성 폭증(5.1→6.5)이 다수결을 희석 — 논문화 진행 중
- 동일 설정 재실행 노이즈 ±0.48%p 실측 → 사전등록·반복 관측 평균·±1%p 규율로 운영
- 재현 절차는 위 영문 섹션의 명령 3개 또는 `docs/final-pipeline.md`(최종일 런북) 참조

## 사용 데이터

- 대회 제공 데이터 (train 17,000 / leaderboard_filtered 831 — 운영진 공지의 오류 문항 627개는 학습·검증에서 제외)
- [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) (Apache 2.0, 무료 공개) — 정수 답 부분집합을 SFT 혼합 학습에 사용 (`remote/prep_numina.py`로 추출, 검증 세트와 문항 중복 제거)
- 자체 생성 데이터 ①: 베이스 모델의 Rejection Sampling 풀이 약 36,000개 (`remote/generate_rft.py`) — **원천 문제는 대회 train 한정**
- 자체 생성 데이터 ②: 강한 교사 모델(Claude, 상용 API)이 작성한 모범 풀이 683개 (`api/gen_teacher_*.py`) —
  **학습 데이터 생성 목적으로만 사용(규칙 5.3.a 근거)이며, 최종 제출 모델에는 미사용** (교사 증류 실험 exp53은 성능 하락으로 기각)

### 데이터 출처 준수 확인 (2026-08-19 감사)

**증류·RFT의 원천 문제는 전량 대회 train 한정이다. test/leaderboard 문제는 어떤 외부 API·서비스에도
입력된 적이 없다** (규칙 5.1.b 및 5.3.b/c 준수). 검증 근거:
- 교사 대상 목록 전 행이 train ID: `data/hard_problems.csv` 2,967행 / `data/hard_problems_top.csv` 1,200행 /
  `data/desktop{,_top}/gold.csv` — **모두 `train-*` 접두** (leaderboard는 `val-*` 접두라 혼입 시 즉시 식별됨)
- RFT 입력: `remote/generate_rft.py`가 `deep_chal_math_train.csv`만 읽음
- `api/` 디렉토리 전체에 leaderboard/test 참조 0건 (전수 grep)
- 로컬 검증셋 483문항은 train에서 seed 고정 분할 (gold 라벨 보유가 그 증거)
- 최종 제출 모델 = **베이스 원본(Qwen2.5-3B-Instruct), 어댑터·추가 학습 미적용** — 학습된 가중치
  제출물이 없는 것이 정상임 (실험용 어댑터는 별도 보관, 추론 시 외부 API·도구·인터넷 미사용)
