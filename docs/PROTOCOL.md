# VRPRM Protocol Reference

Single source of truth for every threshold, mode, and metric convention used
by the paper and this repository. Paper text and supplementary material must
match this document; when an experiment changes a value here, update both.

## 1. Label discretization

| Stage | tau- (negative) | tau+ (positive) | uncertain handling |
| --- | --- | --- | --- |
| SFT data (teacher rollout + multiturn build) | 0.125 | 0.75 | kept in dialogue history, `loss=false` |
| RL data (filtered source cases) | 0.125 | **0.875** | **whole example discarded** if ANY step is uncertain |

RL labels must be near-noise-free because every wrong label directly enters
the reward; SFT tolerates slightly noisier positives because uncertain turns
are masked. **Open item (revision plan):** unifying SFT tau+ to 0.875 is
planned; if adopted, change `data_pipeline/run_all.sh` and rebuild — the
builder masks newly uncertain steps automatically, trainable labels will drop
below 5,592, and all SFT/RL numbers must be re-run.

## 2. SFT data (1,378 conversations / 5,592 trainable labels)

- Source: VisualPRM400K-v1.1-Raw annotations; questions/images from MMPR
  v1.1 (NOT from the five VPB test benchmarks — see §7).
- Teacher: Kimi K2.6 served locally as a W4A8-quantized OpenAI-compatible
  endpoint (`--model-name kimi-k26-w4a8`). Two calls per source example:
  1. hidden-reasoning analysis + strict `{"Score": [...]}` JSON; accepted only
     if parseable, length-matched, and agreeing with the conservative MC label
     on every confident step;
  2. deterministic compression into ONE visible `<think>...</think>` block,
     <=180 words, 100–2,000 chars, else rejected.
- Serialization: system / user(full context incl. reference answer) /
  assistant(think, loss_scale 0.5) / per-step user+assistant(0|1, loss_scale
  2.0; uncertain steps loss=false). Target trainable negative ratio 35%.
- Distillation cost accounting (paper appendix, planned): candidate pool,
  accepted counts, retry/rejection breakdown, token volume, wall time.

## 3. RL (40K examples, 600 steps)

Effective configuration of `easyr1/examples/vrprm/train_vrprm_rl.sh`
(this section is what the supplementary "Implementation Details" must state):

- Data: 40K train / 2K val from the fully confident pool; balanced sampling
  (target negative source ratio 0.50, negative step ratio 0.32,
  hard-mixed bonus 0.03, seed 42).
- Optimization: AdamW bf16, **lr 5e-9**, weight decay 1e-2, **600 steps,
  1 epoch**, GSPO-token loss (clip low 3e-4 / high 4e-4), GRPO group-relative
  advantage, **KL as a loss term** (`use_kl_loss=true`, `low_var_kl`,
  **coef 0.2**; reward-KL disabled), no online filtering.
- Rollouts: n=16 per prompt, 32 prompts per rollout batch (= **512
  generations per optimization step**), temperature 0.8, top-p 0.95, prompt
  cap 8192 / response cap 2048, `guided_regex` enforces
  `<think>...</think><answer>Step i: 0/1 ...</answer>`.
- LoRA: rank 16 / alpha 32 on all-linear modules excluding visual; vision
  tower frozen.
- Reward (`reward_source_macro.py`): 0.97·step + 0.02·format + 0.01·think;
  step = 0.85·min(source macro-F1, Rcount) + 0.15·Rcount;
  Rcount = max(0, 1 − (extra+missing)/n); think full credit in [80, 1200]
  chars, linear ramp below, linear decay above, zero without a think block.
- **Checkpoint selection: peak RL validation overall reward ONLY (step 150
  in the released run).** Any evaluation on VisualProcessBench-derived
  subsets must never inform selection — the mini-VPB dev set (sampled from
  the VPB test split) has been removed from the repository for this reason.

## 4. VisualProcessBench evaluation

- Official protocol: **no reference answer in the model prompt**
  (`VPB_USE_REFERENCE_ANSWER=0`); answers are used only to compute metrics.
  The with-reference variants are oracle diagnostics only.
- Modes: `global_think` (two-stage official), `no_think` (single-pass JSON by
  default; `VPB_NO_THINK_MODE=stepwise` = one request per step),
  `base_warmup` (process-untrained thinking base).
- Step decisions: guided single-token 0/1 at temperature 0; judgment history
  is fed forward turn by turn.
- **VisualPRM-8B baseline**: re-evaluated with its native
  `generate_steps_with_soft_score`, step correct iff soft score > **0.85
  (official threshold)**. Small deviations from the originally published
  numbers (62.18 vs 62.0 overall) come from this re-evaluation.

## 5. Metrics — single Overall column (official pooled protocol)

The paper reports ONE Overall column: `overall_step_macro_f1`, the official
step-pooled macro F1, aligned with the VisualPRM reference paper ("the
overall score is the micro average of the score from different data
sources"). `metrics.py` also emits `mean_source_macro_f1` (unweighted subset
mean) as an internal diagnostic only — it must not appear as a second paper
column; the per-subset mean is derivable from the per-subset columns anyway.
Per-source sample counts (2,866 total):

| Source | Samples | Steps (pos/neg) |
| --- | --- | --- |
| MathVerse | 1,026 | 5,767 / 2,960 |
| MathVision | 712 | 3,875 / 2,350 |
| DynaMath | 570 | 3,469 / 1,091 |
| WeMath | 291 | 1,776 / 582 |
| MMMU | 267 | 1,698 / 708 |

- `overall_step_macro_f1` — all steps pooled across sources, macro F1 over
  correct/incorrect. **Official** protocol (source paper: "overall score is
  the micro average of the score from different data sources");
  step-weighted, so large sources dominate.
- `mean_source_macro_f1` — unweighted mean of the five per-source macro F1s.
  Recomputable for any baseline from published per-source numbers.

The two differ by up to ±2.6 points depending on where a model wins, which
is why the pooled definition is stated explicitly in the paper and the
per-source counts are disclosed in the Supplementary Material.
`analysis/make_paper_table2_vpb.py` emits the single Overall column and
prints the subset mean as a console diagnostic.

## 6. Best-of-N evaluation

- Policies: InternVL2.5-8B/26B/38B, temperature 0.7, ROLLOUT_N=128 cached
  rollouts per sample; BoN selection averages per-step rewards.
- Entries: `run_bon_vrprm.sh` (VRPRM critic), `run_bon_visualprm.sh`
  (VisualPRM critic), pass@K / majority baselines via
  `run_eval_pass_major_*`.
- Paper tables: Table 4 via `analysis/make_paper_table4_bon.py`.
  **Before freezing numbers:** regenerate the incomplete InternVL2.5-38B
  caches (MathVerse-VO 661/788 observed; WeMath/LogicVista missing) and
  re-derive the table.

## 7. Train/eval data provenance (contamination statement)

- VisualPRM400K (both SFT and RL data): questions/images from MMPR v1.1,
  solutions sampled by InternVL2.5 — dataset-level disjoint from VPB.
- VisualProcessBench: questions from the test splits of MMMU / MathVision /
  MathVerse / DynaMath / WeMath, solutions from GPT-4o / Claude / Gemini /
  QvQ / InternVL2.5-78B, human step annotations.
- No direct item-level overlap is expected, but domain kinship between MMPR
  and the five benchmarks exists; a question-text n-gram + image-hash dedup
  check between the 41.4K training sources and VPB's 2,866 items is planned
  (revision plan) and should be reported in the appendix.

## 8. Paper-table generators

| Paper table | Script |
| --- | --- |
| Table 2 (VPB macro F1, both Overalls) | `analysis/make_paper_table2_vpb.py` |
| Table 3 (F1 + latency/tokens) | `analysis/make_paper_table3_cost.py` |
| Table 4 (Bo8 across six benchmarks) | `analysis/make_paper_table4_bon.py` |
