# VRPRM

**VRPRM** is a family of visual reasoning process reward models (PRMs) that
judge the correctness of *intermediate solution steps* for multimodal math and
reasoning problems: Qwen3-VL-8B-Thinking based thinking PRMs built with
rollout-filtered CoT SFT data (distilled from Kimi K2.6) followed by
process-supervised GSPO/GRPO LoRA RL, evaluated on VisualProcessBench
step-level F1 and VLMEval best-of-N selection.

**All thresholds, evaluation modes, metric conventions, and effective training
hyperparameters are defined once in [`docs/PROTOCOL.md`](docs/PROTOCOL.md).**
Paper text must stay aligned with that document.

## Repository layout (paper pipeline first)

```text
data_pipeline/     SFT data generation (teacher rollout -> multiturn build)
  rollout_teacher.py                    stage 1: Kimi two-call rollout + MC-label agreement filter
  build_multiturn.py                    stage 2: evaluation-aligned multiturn builder (--no-think for the M5 ablation)
  build_noncot_sft_from_rl.py           M3 control: 40K non-CoT SFT dataset from the RL split
  subsample_sft.py                      M3 SFT-scale curve (25/50/100%)
  run_all.sh                            one driver, paper defaults
sft/
  train_vrprm_sft.sh                    paper SFT (Qwen3-VL-8B-Thinking, LoRA, weighted loss)
  train_vrprm_sft_nothink.sh            M5 ablation: retrain without the think turn
  train_vrprm_sft_40k_noncot.sh         M3/M2 control: SFT on the same 40K non-CoT examples
  merge_lora.sh                         merge LoRA -> HF for vLLM
easyr1/examples/vrprm/
  train_vrprm_rl.sh                     paper RL run (flattened; defaults ARE the paper config)
  train_vrprm_rl_from_base.sh           M3 ablation: RL without CoT cold start
  reward_source_macro.py                0.97*step + 0.02*format + 0.01*think reward
  vrprm_source_macro.jinja              rollout prompt template
benchmarks/
  visualprocessbench/                   run_eval_vpb.sh = single eval entry (3 modes); cost/
  vlmeval_bon/                          run_bon_vrprm.sh / run_bon_visualprm.sh BoN pipelines
analysis/           paper-table generators (Tables 2/3/4) + training curves
docs/               PROTOCOL.md, DATA_GENERATION.md, REVISION_PLAN.md
easyr1/             EasyR1 fork (GSPO/GRPO trainer) — modified, Apache-2.0
ms-swift/           pinned upstream ms-swift (unmodified) — candidate for submodule
```

Everything removed during consolidation (legacy entrypoints, exploration
launchers, mini-VPB tooling) is preserved on branch
**`archive/snapshot-202609`**.

## Quick start (paper pipeline)

```bash
# 0) raw data: VisualPRM400K-v1.1-Raw (annotations + images) under data/
# 1) SFT data (needs a Kimi K2.6 OpenAI-compatible endpoint):
bash data_pipeline/run_all.sh
# 2) SFT + merge:
bash sft/train_vrprm_sft.sh
bash sft/merge_lora.sh <checkpoint-dir>
# 3) RL (auto-prepares the 40K RL dataset):
bash easyr1/examples/vrprm/train_vrprm_rl.sh
# 4) VisualProcessBench eval (serve the merged model with vLLM first):
cd benchmarks/visualprocessbench
bash serve_vllm.sh /path/to/merged-model
VPB_MODE=global_think bash run_eval_vpb.sh     # writes both Overall metrics
# 5) BoN:
cd ../vlmeval_bon
bash run_generate_rollouts_cache.sh && bash run_bon_vrprm.sh
```

## Published data products (Hugging Face)

Final SFT dataset (`neg35`): **1,378 conversations / 5,592 trainable steps /
34.94% negative steps**, plus accepted/rejected teacher rollouts — see
`docs/DATA_GENERATION.md` and the sibling `huggingface_dataset/` folder.

## Not included

Model checkpoints, generated datasets, rollout caches, wandb logs, and
evaluation outputs are runtime artifacts (`.gitignore`) — regenerate with the
pipelines above.

## License

Apache License 2.0. Vendored forks keep their original licenses.
