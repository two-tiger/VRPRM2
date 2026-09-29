# Qwen3-VL VisualPRM SFT

This folder contains the ms-swift SFT setup for training a visual reasoning process reward model from:

```text
VRPRM_v2.0/rollout_outputs/rollout_sft_success_0615.json
VRPRM_v2.0/rollout_outputs/rollout_sft_success_0614.json
```

By default `train_qwen3vl8b_visualprm_sft.sh` merges them into:

```text
VRPRM_v2.0/rollout_outputs/rollout_sft_success_0614_0615.json
```

The dataset is already in ms-swift standard multimodal format:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "<image>..."}, {"role": "assistant", "content": "..."}], "images": ["/path/to/image.png"]}
```

Run:

```bash
cd <path-to>/VRPRM_v2.0/..
bash VRPRM_v2.0/sft/train_qwen3vl8b_visualprm_sft.sh
```

The default training mode follows `examples/train/multimodal/lora_llm_full_vit/sft.sh`:

- `tuner_type=lora_llm`
- LLM uses LoRA
- ViT and aligner are trained full-parameter with `vit_lr=1e-5` and `aligner_lr=1e-5`
- `num_train_epochs=3`
- `lora_rank=16`
- `learning_rate=1e-4`
- `CUDA_VISIBLE_DEVICES=0`, `NPROC_PER_NODE=1`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=4`
- SFT target is the assistant process-reward output containing preliminary thinking, step analysis, scores, and final judge

Before training, edit `env.list` for GPU count, output path, sequence length, and memory-related settings.

## Train Qwen3-VL-8B With Explicit Think Prompt

Use this script when training the regular Qwen3-VL-8B/Instruct model to visibly output `<think>...</think>` because the system prompt explicitly asks for it.

```bash
cd <path-to>/VRPRM_v2.0/..
bash VRPRM_v2.0/sft/train_qwen3vl8b_explicit_think_visualprm_sft.sh
```

Default dataset:

```text
VRPRM_v2.0/rollout_outputs/visualprm400k_pos_neg_rollout_sft_success_shuffled_with_system_think.json
```

The script validates both conditions before training:

- the system prompt contains an explicit `<think>...</think>` output requirement
- the assistant target starts with `<think>`

Dedicated variables:

| Variable | Default | Description |
| --- | --- | --- |
| `EXPLICIT_THINK_MODEL_PATH` | `MODEL_PATH` from `env.list`, otherwise Qwen3-VL-8B-Instruct snapshot | Base Qwen3-VL-8B model. |
| `EXPLICIT_THINK_DATASET_PATH` | backup dataset above | Dataset passed to ms-swift. |
| `EXPLICIT_THINK_OUTPUT_DIR` | `VRPRM_v2.0/sft/output/qwen3_vl_8b_explicit_think_visualprm_sft` | Training output directory. |
| `REQUIRE_SYSTEM_THINK_PROMPT` | `true` | Validate system prompt explicitly mentions `<think>...</think>`. |
| `REQUIRE_THINK_PREFIX` | `true` | Validate every assistant turn starts with `<think>`. |
| `USE_TORCHRUN` | `false` | Keep single-GPU runs out of torchrun/NCCL. Set `true` for multi-GPU. |

## Train Qwen3-VL-8B-Thinking

Use this path when the SFT data assistant turn starts directly with Qwen3's thinking marker:

```text
<think>...</think>
[Step Analysis]
...
[Final Scores]
{"Score": [...]}
[Final Judgment]
{"Judge": 0 or 1}
```

The marker expected by Qwen3/ms-swift is `<think>`, not `<thinking>`. If a dataset uses `<thinking>`, convert it before training.

Generate positive and negative rollout data separately with the polarity rollout script:

```bash
cd <path-to>/VRPRM_v2.0/..

python VRPRM_v2.0/rollout_visualprm_polarity_sft_data_pipeline.py \
  --polarity positive \
  --assistant-format thinking \
  --score-threshold 0.25 \
  --sample-total 10000 \
  --num-workers 32

python VRPRM_v2.0/rollout_visualprm_polarity_sft_data_pipeline.py \
  --polarity negative \
  --assistant-format thinking \
  --score-threshold 0.25 \
  --sample-total 10000 \
  --num-workers 32
```

The rollout script binarizes VisualPRM400K MC scores with the same threshold before validation: `score > 0.25` becomes `1`, and `score <= 0.25` becomes `0`.
It also tries to disable hidden API thinking by default with `--api-extra-body-json '{"chat_template_kwargs":{"enable_thinking":false}}'` and asks the rollout model to write focused visible problem reasoning in `<think>...</think>`, capped at 200 words by the prompt.

The thinking SFT script now expects a single pre-merged and shuffled dataset by default:

```text
VRPRM_v2.0/rollout_outputs/visualprm400k_pos_neg_rollout_sft_success_shuffled.json
```

Then train the thinking model:

```bash
cd <path-to>/VRPRM_v2.0/..
bash VRPRM_v2.0/sft/train_qwen3vl8b_thinking_visualprm_sft.sh
```

The default thinking model path is:

```text
Qwen/Qwen3-VL-8B-Thinking (Hub id or local snapshot)
```

The script uses dedicated `THINKING_*` environment variables so the regular Instruct-model `env.list` does not override the thinking run:

| Variable | Default | Description |
| --- | --- | --- |
| `THINKING_MODEL_PATH` | `Qwen/Qwen3-VL-8B-Thinking (Hub id or local snapshot)` | Base thinking model. |
| `THINKING_DATASET_PATH` | `VRPRM_v2.0/rollout_outputs/visualprm400k_pos_neg_rollout_sft_success_shuffled.json` | Dataset passed to ms-swift. |
| `THINKING_OUTPUT_DIR` | `VRPRM_v2.0/sft/output/qwen3_vl_8b_thinking_visualprm_sft` | Training output directory. |
| `REQUIRE_THINK_PREFIX` | `true` | Validate every assistant turn starts with `<think>`. |
| `USE_TORCHRUN` | `false` | Keep single-GPU runs out of torchrun/NCCL. Set `true` for multi-GPU. |
| `NPROC_PER_NODE` | unset | Number of local processes when `USE_TORCHRUN=true`. |

Thinking-specific ms-swift settings in the script:

```text
--preserve_thinking true
--add_non_thinking_prefix false
```

These preserve assistant thinking content during training and prevent ms-swift from adding a non-thinking prefix to samples.

## Train Global-Thinking Stepwise Multi-Turn SFT

For the `api_eval_global_think_stepwise.py` evaluation flow, build one multi-turn conversation per original VisualPRM sample:

```bash
cd <path-to>/VRPRM_v2.0/..
python VRPRM_v2.0/build_global_think_stepwise_multiturn_sft_dataset.py
```

Default output:

```text
VRPRM_v2.0/rollout_outputs/visualprm400k_global_think_stepwise_multiturn_sft_neg35.json
```

The builder keeps all negative rollout samples, samples positive rollout samples until the trainable step-level negative ratio is about `35%`, and creates a single `messages` conversation per source sample:

```text
system
user: full question/reference/candidate solution with image
assistant: <think>global thinking / step overview</think>        loss_scale=0.5
user: current step 0 + previous judgments
assistant: 0 or 1                                               loss_scale=2.0
...
```

Middle-score uncertain steps are kept in the conversation history with `loss=false`, so later step context stays continuous without training on low-confidence labels.

Train Qwen3-VL-8B-Thinking on this multi-turn data:

```bash
bash VRPRM_v2.0/sft/train_qwen3vl8b_thinking_global_stepwise_multiturn_sft.sh
```

This script sets:

```text
--loss_scale default
--is_binary_loss_scale false
--preserve_thinking true
--add_non_thinking_prefix false
```

`--is_binary_loss_scale false` is required because the dataset uses assistant-level `loss_scale=0.5` and `loss_scale=2.0`.

## Merge checkpoint to Hugging Face format

The SFT script trains Qwen3-VL-8B in `lora_llm` mode: the LLM is saved as LoRA adapter weights, while the ViT/aligner
weights are saved with the checkpoint. To export a standalone Hugging Face model directory, merge the checkpoint with
ms-swift:

```bash
cd <path-to>/VRPRM_v2.0/..
bash VRPRM_v2.0/sft/merge_qwen3vl8b_visualprm_sft.sh
```

By default this finds the latest adapter checkpoint under `OUTPUT_DIR` and writes a merged HF model to
`<checkpoint>-merged`.

For example, with the current default output directory this resolves to:

```text
input : VRPRM_v2.0/sft/output/qwen3_vl_8b_visualprm_sft/v2-20260618-174705/checkpoint-3438
output: VRPRM_v2.0/sft/output/qwen3_vl_8b_visualprm_sft/v2-20260618-174705/checkpoint-3438-merged
```

To choose paths explicitly, pass the checkpoint path and output path:

```bash
bash VRPRM_v2.0/sft/merge_qwen3vl8b_visualprm_sft.sh \
  VRPRM_v2.0/sft/output/qwen3_vl_8b_visualprm_sft/v2-20260618-174705/checkpoint-3438 \
  VRPRM_v2.0/sft/output/qwen3_vl_8b_visualprm_sft/qwen3_vl_8b_visualprm_sft_hf
```

The script is a wrapper around:

```bash
python VRPRM_v2.0/ms-swift/swift/cli/main.py export \
  --adapters <checkpoint> \
  --merge_lora true \
  --output_dir <merged_hf_dir> \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --safe_serialization true \
  --max_shard_size 5GB
```

Useful environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `CKPT_DIR` | latest adapter checkpoint under `OUTPUT_DIR` | Checkpoint to merge. Positional arg 1 takes precedence. |
| `MERGED_OUTPUT_DIR` | `<CKPT_DIR>-merged` | Output HF model directory. Positional arg 2 takes precedence. |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU used for merge. |
| `TORCH_DTYPE` | `bfloat16` | Dtype used when loading/saving the merged model. |
| `MAX_SHARD_SIZE` | `5GB` | Max size of each saved safetensors shard. |
| `MERGE_DEVICE_MAP` | unset | Optional `device_map`, for example `auto` or `cpu`. |
| `MAX_MEMORY` | unset | Optional memory map used with `device_map=auto`. |
| `EXIST_OK` | `false` | Set to `true` if the output directory already exists. |
| `DRY_RUN` | `false` | Set to `true` to print the resolved command without loading the model. |

Examples:

```bash
# Print the resolved command only.
DRY_RUN=true bash VRPRM_v2.0/sft/merge_qwen3vl8b_visualprm_sft.sh

# Merge a specific checkpoint and allow an existing output directory.
EXIST_OK=true bash VRPRM_v2.0/sft/merge_qwen3vl8b_visualprm_sft.sh \
  VRPRM_v2.0/sft/output/qwen3_vl_8b_visualprm_sft/v2-20260618-174705/checkpoint-3438 \
  VRPRM_v2.0/sft/output/qwen3_vl_8b_visualprm_sft/checkpoint-3438-hf

# Use automatic device placement when one GPU is not enough.
MERGE_DEVICE_MAP=auto \
MAX_MEMORY='{0: "70GiB", "cpu": "120GiB"}' \
bash VRPRM_v2.0/sft/merge_qwen3vl8b_visualprm_sft.sh
```

The merged output should contain a normal Hugging Face model layout, including files such as `config.json`,
`model.safetensors.index.json`, `model-*.safetensors`, tokenizer files, and processor/preprocessor configs. Use the
merged directory as `model_id_or_path` for Transformers, vLLM, or other Hugging Face-compatible inference stacks.
