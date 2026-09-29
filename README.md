# VRPRM v2.0

**VRPRM** is a family of visual reasoning process reward models (PRMs) that judge the
correctness of *intermediate solution steps* for multimodal math and reasoning
problems. Version 2.0 builds Qwen3-VL-8B based thinking PRMs with:

- **Rollout-filtered SFT data** generated from VisualPRM400K raw Monte-Carlo process
  labels, including global-thinking + stepwise multi-turn formats.
- **Process-supervised RL** (GSPO/GRPO LoRA) on top of the SFT model with step-level
  reward functions.
- **Evaluation** on VisualProcessBench step-level F1 and VLMEval best-of-N selection.

## Repository Structure

```text
VRPRM_v2.0/
├── rollout_*_sft_data_pipeline.py   # Rollout SFT data generation pipelines
├── build_global_think_stepwise_multiturn_sft_dataset.py
├── docs/DATA_GENERATION.md          # Detailed data pipeline documentation
├── sft/                             # ms-swift SFT launch scripts + docs
├── easyr1/                          # EasyR1 fork with VisualPRM GSPO/GRPO training
│   ├── examples/                    #   RL launch scripts and reward functions
│   ├── scripts/                     #   Dataset preparation helpers
│   └── verl/                        #   Modified trainer/rollout utilities
├── ms-swift/                        # Pinned ms-swift used for SFT
├── benchmarks/
│   ├── visualprocessbench/          # Step-level PRM evaluation (macro F1)
│   └── vlmeval_bon/                 # Best-of-N selection on VLMEval datasets
└── figures/                         # Training curves and plotting script
```

## Quick Start

### 1. Prepare raw data

Download `VisualPRM400K-v1.1-Raw` (annotations + images) from Hugging Face into:

```text
data/VisualPRM400K-v1.1-Raw/
```

The `data/`, `rollout_outputs/`, `sft/output/`, and `easyr1/data/` directories are
generated at runtime and are not committed.

### 2. Generate rollout SFT data

Serve a strong multimodal model behind an OpenAI-compatible endpoint, then run the
rollout pipelines. See [docs/DATA_GENERATION.md](docs/DATA_GENERATION.md) for full
details:

```bash
python rollout_visualprm_global_think_stepwise_sft_data_pipeline.py --polarity negative
python rollout_visualprm_global_think_stepwise_sft_data_pipeline.py --polarity positive
python build_global_think_stepwise_multiturn_sft_dataset.py
```

Endpoint defaults are `http://127.0.0.1:8000/v1` and can be overridden with
`OPENAI_BASE_URL` / `OPENAI_API_KEY`.

### 3. SFT

```bash
bash sft/train_qwen3vl8b_thinking_global_stepwise_multiturn_sft.sh
bash sft/merge_qwen3vl8b_visualprm_sft.sh   # merge LoRA -> HF format
```

See [sft/README.md](sft/README.md) for all training variants and environment variables.

### 4. RL

RL uses the vendored EasyR1 fork with process reward functions under
`easyr1/examples/reward_function/`:

```bash
bash easyr1/examples/qwen3_vl_8b_global_stepwise_visualprm400k_gspo_lora.sh
```

Each launcher auto-prepares its RL dataset from the raw VisualPRM400K annotations;
model/data paths are configurable through environment variables (`MODEL_PATH`,
`DATA_ROOT`, ...).

### 5. Evaluation

**VisualProcessBench (step-level macro F1):**

```bash
cd benchmarks/visualprocessbench
bash run_eval_no_think_stepwise.sh        # or the global-think-stepwise variant
```

> The benchmark ships `test.jsonl`, but the `images/` assets must be downloaded
> separately (they are not committed). Point or link
> `VisualProcessBench/images` to the downloaded asset directory before running.

**VLMEval best-of-N:**

```bash
cd benchmarks/vlmeval_bon
# generate rollouts, score with the PRM, select BoN, then evaluate
bash run_generate_rollouts_cache.sh
bash run_score_rollouts_cache.sh
bash run_eval_bon_from_cache.sh
```

See [benchmarks/vlmeval_bon/README.md](benchmarks/vlmeval_bon/README.md). Rollout
caches and downloaded VLMEval TSVs are runtime artifacts and are not committed.

## Training Curves

SFT/RL reward and accuracy curves are available under
[`figures/training_curves/`](figures/training_curves/) (`sft_training_curves.svg`,
`rl_training_curves.svg`, `sft_rl_training_summary.svg`), together with the plotting
script and source CSVs.

## Published Data Products (Hugging Face Dataset)

The generated SFT data products are **not committed to this repository**. They are
prepared for release as a Hugging Face Dataset in the sibling folder:

```text
../huggingface_dataset/
├── README.md       # HF dataset card (metadata, formats, statistics, usage)
├── LICENSE
└── data/
    ├── visualprm400k_positive_global_think_stepwise_sft_success.json   (58M, 12,571 rows)
    ├── visualprm400k_positive_global_think_stepwise_sft_failed.json    (15M, 1,237 rows)
    ├── visualprm400k_negative_global_think_stepwise_sft_success.json   (15M, 3,555 rows)
    ├── visualprm400k_negative_global_think_stepwise_sft_failed.json    (27M, 2,090 rows)
    ├── visualprm400k_global_think_stepwise_multiturn_sft_neg35.json    (9.1M, final dataset)
    └── visualprm400k_global_think_stepwise_multiturn_sft_neg35.stats.json
```

Upload it with:

```bash
huggingface-cli upload <org>/VRPRM2-GlobalThinkStepwise-SFT \
  huggingface_dataset . --repo-type dataset
```

Final dataset (`neg35`): **1,378 samples / 5,592 trainable steps / 34.94% negative
steps / 18,616 messages**.

## Vendored Forks

- [`easyr1/`](easyr1/): fork of [hiyouga/EasyR1](https://github.com/hiyouga/EasyR1) with
  VisualPRM-specific reward functions, prompt templates, launch scripts, and small
  trainer/rollout modifications. Base commit: `dd71bbd`.
- [`ms-swift/`](ms-swift/): pinned upstream
  [modelscope/ms-swift](https://github.com/modelscope/ms-swift) (no local
  modifications) used by the SFT scripts. Base commit: `e13595e3c`.

Both keep their original Apache-2.0 licenses.

## Not Included

Model checkpoints, generated datasets, wandb logs, and evaluation outputs are not part
of this repository. Regenerate them with the pipelines above, or restore them from
your internal release archive.

## License

This project is released under the [Apache License 2.0](LICENSE).

## Citation

If you use VRPRM, please cite:

```bibtex
@misc{vrpm2026,
  title  = {VRPRM: Visual Reasoning Process Reward Models},
  author = {VRPRM Team},
  year   = {2026},
  note   = {Preprint}
}
```
