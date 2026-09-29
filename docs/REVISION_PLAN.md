# VRPRM Revision Plan (consolidated after repo reorganization)

Status: 2026-09-29. Supersedes the earlier M1–M7 list. Repository entry points
referenced below are the post-reorganization names (see README.md); every
protocol value lives in docs/PROTOCOL.md and must stay in sync with the paper.

Legend: ☐ todo · ◐ in progress · ☑ done

---

## Phase 0 — Completed (code & paper text, no GPU needed)

- ☑ N1 mini-VPB (test-derived dev set) and checkpoint sweep removed from the
  repo; checkpoint selection pinned to RL validation reward only. Supplementary
  D now states this is the **sole** selection signal.
- ☑ N2 Supplementary C "RL Stage" rewritten with the actual configuration
  (lr 5e-9, KL loss coef 0.2 low-var, 600 steps/1 epoch, 32×16=512
  generations/step, GSPO-token clip 3e-4–4e-4, LoRA 16/32, temp 0.8,
  guided decoding). Main-text "KL penalty" phrasing aligned.
- ☑ N4 Main text now states: Overall = official pooled protocol; reference
  answer never shown to the model (official protocol); VisualPRM-8B
  re-evaluated with official code and threshold 0.85 (deviations stem from
  re-evaluation). Supplementary gains a "Metric Conventions" subsection with
  per-source counts and the unweighted-mean convention. Page limit preserved
  (8 pages, references-only final page).
- ☑ Repo reorganized on `main`; pre-reorganization snapshot on
  `archive/snapshot-202609`.

## Phase 1 — Metric & table restructure (P0, analysis-only, no retraining)

- ☐ R1 Dual-Overall reporting in Table 2: pooled (official) **and** subset
  mean columns; per-source sample counts in the caption; baselines marked
  quoted-vs-re-evaluated. Tool: `analysis/make_paper_table2_vpb.py`
  (emits both). Verify Athena-PRM "#Samples 155K" provenance (its arXiv says
  ~5K process labels + 600K ORM init; footnote the ORM accounting).
- ☐ R2 Add a "Base Model" column to Table 2 (Qwen3-VL-8B-Thinking /
  InternVL2.5-8B / Qwen2.5-VL-7B / …) to make cross-base comparisons explicit.
- ☐ R3 Narrative shift: flagship claim moves from "45.5K beats 400K"
  (cross-base) to (a) BoN same-candidate-pool critic comparison
  (+21.77 vs +6.12, fully controlled) and (b) same-base per-label efficiency
  (Phase 2 controls). Table 2 stays as reference with the base column.
- ☐ R4 Honest per-subset discussion: VRPRM wins MathVerse/MathVision (the two
  largest sources); Athena-PRM leads DynaMath/MMMU/WeMath (three smallest);
  discuss the MMMU gap (~10 pts, knowledge-heavy steps).

## Phase 2 — Attribution controls (P0, training required)

- ☐ C1 `sft/train_vrprm_sft_40k_noncot.sh` — base → 40K non-CoT SFT.
  Dataset via `data_pipeline/build_noncot_sft_from_rl.py`.
- ☐ C2 CoT-cold-start → 40K non-CoT SFT (same data & init as VRPRM-RL, SFT
  instead of RL). Add a thin wrapper like train_vrprm_sft_40k_noncot.sh with
  `THINKING_MULTITURN_DATASET_PATH` pointed at the cold-start merged model +
  non-CoT data; this is the cleanest "RL vs SFT" attribution.
- ☐ C3 `easyr1/examples/vrprm/train_vrprm_rl_from_base.sh` — RL without cold
  start (justifies stage 1; watch the format reward early in training).
- ☐ N3 w/o-Thinking ablation matrix (accepted design):
  | Variant | Training | Inference interface | Question answered |
  | --- | --- | --- | --- |
  | VRPRM (existing) | CoT SFT+RL | global think + stepwise | main result |
  | w/o Think (skip) | same ckpt | single_pass **and** stepwise (run both) | value of generated rationale at test time |
  | w/o Think (retrained) | `train_vrprm_sft_nothink.sh` (same 5,592 labels) | stepwise | value of the CoT **training signal** |
  | 40K non-CoT SFT | C1 | stepwise | data vs recipe |
  Rename table rows to "thinking skipped at inference" vs "retrained w/o
  thinking"; fix the Table 2/3 caption ("retaining stepwise scoring" only
  holds for the stepwise run). `run_eval_vpb.sh VPB_MODE=no_think
  VPB_NO_THINK_MODE=stepwise` provides the missing interface run.
- ☐ S1 SFT-scale curve: `data_pipeline/subsample_sft.py` at 25/50/100%,
  three `train_vrprm_sft.sh` runs (cheap, LoRA).
- ☐ S2 RL-scale curve: `train_vrprm_rl.sh` with `MAX_TRAIN_SAMPLES=10000/20000`
  (default 40000), fixed 600 steps.

## Phase 3 — Variance & integrity (P0, evaluation heavy)

- ☐ V1 ≥3 SFT training seeds (cheap) and ≥2 RL seeds; report mean±std in the
  appendix for VRPRM rows; ensure per-subset numbers are measured
  independently (resolves the suspicious uniform +3.35 across subsets).
- ☐ V2 BoN candidate-pool resampling ×3 (temperature-0.7 pools are the main
  noise source); report mean±std for Table 4 deltas.
- ☐ V3 Complete the InternVL2.5-38B rollout caches (MathVerse-VO 661/788,
  WeMath/LogicVista missing) and regenerate Table 4 via
  `analysis/make_paper_table4_bon.py`; verify every cell traces to a cache.
- ☐ V4 Train/eval dedup check: question-text n-gram + image hash between the
  41.4K training sources and VPB's 2,866 items; report overlap (expect ≈0)
  plus a provenance statement (MMPR v1.1 vs five benchmark test splits) and a
  domain-kinship limitation note.

## Phase 4 — Cost accounting & disclosure (P2)

- ☐ K1 Distillation ledger appendix: candidate pool → accepted 1,378 → Kimi
  call counts (2 calls + retries), rejection breakdown (JSON invalid /
  count mismatch / label disagreement / compression filter), token volume,
  wall time, quantized local serving (kimi-k26-w4a8). Compare against
  VisualPRM's 400K MCTS pipeline and Athena's 1/45 GPU-hour filtering;
  acknowledge inherited MCTS-label generation cost.
- ☐ K2 Optional: unify SFT tau+ 0.75 → 0.875 (paper currently stage-specific
  but now explicitly justified in PROTOCOL.md §1 — decide justify-vs-unify
  before re-running Phase 2; if unified, all SFT/RL numbers re-run once).

## Phase 5 — Writing & scholarship (P2, no compute)

- ☐ W1 Citations: fix OpenCompass (Contributors 2023); add Lightman et al.
  2023 (Let's Verify Step by Step), Math-Shepherd, OmegaPRM, PRIME.
- ☐ W2 Tone down "first-ever"/"Pioneering" (Table 1 marks TIM-PRM as
  multimodal CoT-PRM; the qualifier is "trained by RL").
- ☐ W3 Fix table order (Table 4 physically before Table 3), Figure 2
  declutter, abstract wording ("non-thinking PRMs that use a total of 400K
  data"), 5,592 → "≈5.6K" or exact count, Bo1-anchored proprietary comparison.
- ☐ W4 Decide and apply: think/step loss-weight 1.0/1.0 control run (one SFT)
  to substantiate the "Weighted SFT" contribution claim.

## Deferred (explicit non-goals for this round)

- Full python merge of the four cost .py files (1335+ lines, needs GPU
  verification); current cost/ keeps faithful wrappers.
- ms-swift vendored → submodule/pip pin (repo slimming; no functional change).
- verl fork diff extraction (`VERL_PATCH.md`) — requires cloning upstream
  at dd71bbd on the server.

## Execution order

Phase 1 (analysis only) → C1/C3(no-think retrain)/C3-interface run → V1/V2
(variance) → C2/C3-RL-from-base/S1/S2 (remaining controls) → V3/V4 → Phase 4/5.
All new numbers flow through `analysis/make_paper_table*_*.py`; update
docs/PROTOCOL.md whenever a value changes, then sync the paper text.
