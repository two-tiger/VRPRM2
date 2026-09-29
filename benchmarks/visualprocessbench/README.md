# VisualProcessBench Evaluation

Step-level macro F1 evaluation on VisualProcessBench (VPB): 2,866 samples,
26,950 step labels, labels are `1` correct / `0` neutral / `-1` incorrect
(neutral steps are excluded from metrics). Five sources:
**MathVerse (1,026), MathVision (712), DynaMath (570), WeMath (291), MMMU (267)**.

> `test.jsonl` and `stats.json` are committed. The `images/` directory is a
> local symlink to separately distributed assets — download and link it before
> evaluating (`bash run_prepare.sh`).

## Layout

| File | Purpose |
| --- | --- |
| `metrics.py` | Step-level macro F1. Reports **both** Overall conventions: `overall_step_macro_f1` (step-pooled, official) and `mean_source_macro_f1` (unweighted subset mean). See docs/PROTOCOL.md. |
| `run_eval_vpb.sh` | **Single entry point** for API-served models. `VPB_MODE=global_think` (official two-stage protocol) \| `no_think` (`VPB_NO_THINK_MODE=single_pass\|stepwise`) \| `base_warmup` (process-untrained base model). |
| `eval_vpb_global_think.py` | Two-stage evaluator: one global `<think>` block, then one guided 0/1 single-token request per step with judgment history. |
| `eval_vpb_no_think.py` | No-thinking evaluator (single-pass JSON or stepwise). |
| `vpb_common.py` | Shared client/IO/retry/parsing helpers + the base-model warmup evaluator (`VPB_MODE=base_warmup`). |
| `run_eval_visualprm.sh` → `eval_visualprm_baseline.py` | VisualPRM-8B baseline re-evaluation via its native `generate_steps_with_soft_score`; step is correct when soft score > threshold (**official 0.85**). Runs in the transformers environment, separate from the vLLM/API environment. |
| `serve_vllm.sh` / `serve_vllm_dp.sh` | Single-instance / multi-replica vLLM serving of a merged checkpoint. |
| `prepare_visualprocessbench.py`, `run_prepare.sh` | Benchmark preparation and image symlink. |
| `cost/` | Latency / token-overhead benchmarks (Table 3). See `cost/README.md`. |

## Protocol

1. **Reference answers are NOT sent to the model** (`VPB_USE_REFERENCE_ANSWER=0`,
   the official protocol). VPB answers are used only after prediction to compute
   metrics. `VPB_USE_REFERENCE_ANSWER=1` exists only for oracle-style diagnostics.
2. Model selection / checkpoint selection must never use VPB-derived subsets;
   RL checkpoints are selected by peak RL validation reward only.
3. After each run, `run_eval_vpb.sh` writes `<output>.metrics.json` containing
   both Overall conventions. Report both in papers and state which is which.

## Quick start

```bash
# serve a merged checkpoint (bash sft/merge_lora.sh <ckpt> first), then:
VPB_MODE=global_think VPB_MODEL=auto \
VPB_OUTPUT=outputs/vrprm_rl_global_think_no_ref_predictions.jsonl \
bash run_eval_vpb.sh

# no-thinking ablation (single-pass JSON, the paper's w/o Thinking setting):
VPB_MODE=no_think VPB_NO_THINK_MODE=single_pass bash run_eval_vpb.sh

# VisualPRM-8B baseline (transformers environment, official threshold 0.85):
VISUALPRM_MODEL_PATH=VisualPRM/VisualPRM-8B bash run_eval_visualprm.sh
```

Useful resume/debug options (all modes): `VPB_LIMIT` (smoke test),
`VPB_OVERWRITE=1`, `VPB_RETRY_ERRORS=1`, `VPB_RETRY_INVALID=1`,
`VPB_CONCURRENCY`, `VPB_REQUEST_TIMEOUT`.

## Compute-cost probes

Because API evaluation and VisualPRM local evaluation require different
environments, run them separately — see `cost/README.md`.

## Sources

- Paper: https://arxiv.org/abs/2503.10291
- Pre-reorganization scripts (legacy reference-answer entrypoints, mini-dev
  tooling, original wrapper names) are preserved on the
  `archive/snapshot-202609` branch.
