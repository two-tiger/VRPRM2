# VisualProcessBench Evaluation

This folder organizes VisualProcessBench and provides two evaluation paths:

1. `api_eval_sft.py`: evaluate the SFT visual process reward model trained in `VRPRM_v2.0/sft`. This uses the model as a generative MLLM judge and parses the saved `{"Score": [...]}` JSON.
2. `visualprm_paper_eval.py`: reproduce the VisualPRM paper-style PRM evaluation by judging each step with `+` or `-` and predicting correct when `p(+) - p(-) > threshold`.
3. `api_eval_global_think_stepwise.py`: evaluate a two-stage SFT PRM flow. It first asks the model for one global `<think>...</think>` block, then scores each candidate step with a guided single-token `0/1` request.

Paper facts used here:

- VisualProcessBench has 2,866 samples and 26,950 step labels.
- Labels are `1` correct, `0` neutral, `-1` incorrect.
- The benchmark asks models to detect all erroneous steps, not only the first erroneous step.
- The reported metric is macro F1 over correct and incorrect steps, excluding neutral steps.
- For PRMs, the paper classifies a step as correct when the probability of outputting `+` exceeds that of `-` by a threshold.

Sources:

- Paper: https://arxiv.org/abs/2503.10291
- Reference implementation: VRPRM v1 `VisualProcessBench_PRM` evaluation suite

> `test.jsonl` and `stats.json` are committed. The `images/` directory is a
> local symlink to separately distributed assets and is **not** part of the
> repository — download the assets and link them here before evaluating.

## Prepare Data

```bash
bash VRPRM_v2.0/benchmarks/visualprocessbench/run_prepare.sh
```

This creates:

```text
VRPRM_v2.0/benchmarks/visualprocessbench/VisualProcessBench/test.jsonl
VRPRM_v2.0/benchmarks/visualprocessbench/VisualProcessBench/images -> <downloaded VisualProcessBench assets>/images
VRPRM_v2.0/benchmarks/visualprocessbench/VisualProcessBench/stats.json
```

## Evaluate SFT Model

First serve your trained SFT model with an OpenAI-compatible endpoint. If the model was trained with ms-swift LoRA SFT, merge the LoRA checkpoint into the base model before starting vLLM.

### Merge ms-swift LoRA Checkpoint

Use the checkpoint directory produced by ms-swift, for example `.../checkpoint-35`:

```bash
export BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct (Hub id or local snapshot)
bash VRPRM_v2.0/benchmarks/visualprocessbench/export_merged_sft_for_vllm.sh \
  VRPRM_v2.0/sft/output/qwen3_vl_8b_visualprm_sft/v1-20260614-145603/checkpoint-35
```

By default the merged model is written to:

```text
<checkpoint_dir>-merged
```

If you pass an ms-swift run directory that contains multiple `checkpoint-*` folders, the script selects the latest checkpoint that contains LoRA adapter files by version sort.

You can choose a custom output path:

```bash
export BASE_MODEL=/path/to/base-qwen3-vl-model
export MERGED_MODEL_DIR=/path/to/merged-model
bash VRPRM_v2.0/benchmarks/visualprocessbench/export_merged_sft_for_vllm.sh /path/to/checkpoint
```

The merge script runs `swift export --merge_lora true`, then copies tokenizer and processor files from `BASE_MODEL` so the merged Qwen3-VL model can be loaded by vLLM. If the output directory already exists, the script exits instead of silently reusing stale weights. To recreate it:

```bash
OVERWRITE_MERGED=1 bash VRPRM_v2.0/benchmarks/visualprocessbench/export_merged_sft_for_vllm.sh /path/to/checkpoint
```

### vLLM Manual Deployment

Start vLLM with the merged model directory, a full model directory, or a Hugging Face model id you want to evaluate:

```bash
export MODEL_PATH=/path/to/checkpoint-merged
export CUDA_VISIBLE_DEVICES=0,1
export TENSOR_PARALLEL_SIZE=2
bash VRPRM_v2.0/benchmarks/visualprocessbench/serve_sft_vllm_manual.sh "${MODEL_PATH}"
```

You can also pass the model directly as the first argument:

```bash
bash VRPRM_v2.0/benchmarks/visualprocessbench/serve_sft_vllm_manual.sh /path/to/checkpoint-merged
```

Useful deployment overrides:

```bash
SERVED_MODEL_NAME=qwen3-vl-8b-visualprm-sft \
HOST=0.0.0.0 \
PORT=8000 \
MAX_MODEL_LEN=8192 \
MAX_NUM_SEQS=8 \
GPU_MEMORY_UTILIZATION=0.90 \
LIMIT_MM_PER_PROMPT='{"image": 8}' \
bash VRPRM_v2.0/benchmarks/visualprocessbench/serve_sft_vllm_manual.sh /path/to/checkpoint-merged
```

In another shell:

```bash
export VPB_BASE_URL=http://127.0.0.1:8000/v1
export VPB_API_KEY=EMPTY
export VPB_MODEL=auto
bash VRPRM_v2.0/benchmarks/visualprocessbench/run_eval_sft_api.sh
```

`VPB_MODEL=auto` uses the first model returned by the vLLM `/models` endpoint. If you set `SERVED_MODEL_NAME` when starting vLLM, you can set `VPB_MODEL` to the same value instead.

For a smoke test:

```bash
VPB_LIMIT=10 bash VRPRM_v2.0/benchmarks/visualprocessbench/run_eval_sft_api.sh
```

`run_eval_sft_api.sh` uses stepwise reward extraction by default:

1. Run one short multimodal chat request to collect a concise `<think>...</think>` context.
2. Query each solution step with `completions.create(max_tokens=1)` and vLLM `guided_choice=["1", "0"]`.

This avoids asking the model to emit a long `{"Score": [...]}` array in one response, so long solutions are less likely to produce a mismatched number of step judgments. To use the old full-response parser:

```bash
VPB_REWARD_MODE=full bash VRPRM_v2.0/benchmarks/visualprocessbench/run_eval_sft_api.sh
```

### Global Thinking + Stepwise Scoring

For models trained or prompted with a two-stage format, run:

```bash
python VRPRM_v2.0/benchmarks/visualprocessbench/api_eval_global_think_stepwise.py \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --model auto \
  --output VRPRM_v2.0/benchmarks/visualprocessbench/outputs/qwen3-vl-8b-global-think-stepwise_no_ref_predictions.jsonl \
  --concurrency 4 \
  --think-max-tokens 1024 \
  --use-reference-answer 0
```

This evaluator saves the generated `global_thinking` in each prediction row, then sends one guided `0/1` scoring request per step. By default, reference answers are not included in model prompts; VPB labels are used only after prediction to compute metrics. Set `--use-reference-answer 1` only for an oracle-style diagnostic.

Legacy/oracle entrypoints that preserve the old reference-answer prompt behavior are kept as `api_eval_global_think_stepwise_with_ref_legacy.py` and `run_eval_global_think_stepwise_with_ref_legacy.sh`.

### No-Thinking Ablation

To ablate the global thinking stage while keeping a visual scoring request:

```bash
VPB_BASE_URL=http://127.0.0.1:8000/v1 \
VPB_MODEL=vrprm-v2-sft \
VPB_CONCURRENCY=4 \
VPB_LIMIT=0 \
VPB_NO_THINK_MODE=single_pass \
VPB_MAX_TOKENS=512 \
VPB_USE_REFERENCE_ANSWER=0 \
VPB_OUTPUT=VRPRM_v2.0/benchmarks/visualprocessbench/outputs/vrprm-v2-sft_no_think_single_pass_no_ref_predictions.jsonl \
bash VRPRM_v2.0/benchmarks/visualprocessbench/run_eval_no_think_stepwise.sh
```

This removes the initial global `<think>...</think>` request entirely. By default, each sample is evaluated with one multimodal request: the model receives the image(s), question, and full candidate solution once, then directly outputs `{"Score": [...]}` without chain-of-thought text. This keeps the task visual while ablating explicit global thinking.

Set `VPB_NO_THINK_MODE=stepwise` only to reproduce the older one-request-per-step ablation. In that mode, `VPB_STEP_IMAGE_MODE=each_step` attaches image(s) to every step-scoring request, while `VPB_STEP_IMAGE_MODE=never` makes the step scorer text-only.

Legacy/oracle entrypoints that preserve the old reference-answer prompt behavior are kept as `api_eval_no_think_stepwise_with_ref_legacy.py` and `run_eval_no_think_stepwise_with_ref_legacy.sh`.

### Thinking Ablation Cost Probe

Because VRPRM/SFT API evaluation and VisualPRM local evaluation may require different environments, run them separately. First, in the VRPRM environment with the SFT model already deployed by vLLM:

```bash
SFT_BASE_URL=http://127.0.0.1:8000/v1 \
SFT_MODEL=vrprm-v2-sft \
SFT_CONCURRENCY=4 \
SFT_USE_REFERENCE_ANSWER=0 \
SFT_TOKENIZER_PATH=/path/to/merged-sft-model \
VPB_LIMIT=32 \
VPB_SEED=42 \
VPB_RUN_NAME=think_ablation_cost_test \
VPB_OVERWRITE=1 \
bash VRPRM_v2.0/benchmarks/visualprocessbench/run_benchmark_vrprm_think_cost.sh
```

Then, in the VisualPRM environment, reuse the same selected ids:

```bash
VISUALPRM_MODEL_PATH=<path-or-hub-id-to>/VisualPRM-8B \
VPB_SELECTED_IDS_FILE=VRPRM_v2.0/benchmarks/visualprocessbench/outputs/compute_cost/think_ablation_cost_test/selected_ids.json \
VPB_RUN_NAME=think_ablation_cost_test \
bash VRPRM_v2.0/benchmarks/visualprocessbench/run_benchmark_visualprm_cost_subset.sh
```

The comparison table is written to `outputs/compute_cost/<run-name>/comparison.csv`. The main columns are per-sample latency, generated tokens, and score output units. `avg_generated_tokens_per_sample` is real language-model completion tokens for VRPRM/SFT API calls; VisualPRM follows `run_eval_visualprm_paper.sh` and is non-generative, so generated tokens are `0`. `avg_score_outputs_per_sample` counts produced step scores, so VisualPRM should be close to the average number of steps in the selected subset.

The API evaluator supports resume by default. Each finished sample is appended
to JSONL immediately and flushed to disk, so an interrupted run can continue
from existing ids:

```bash
bash VRPRM_v2.0/benchmarks/visualprocessbench/run_eval_sft_api.sh
```

Useful resume/debug options:

```bash
# Start from scratch and overwrite previous predictions, errors, and status.
VPB_OVERWRITE=1 bash VRPRM_v2.0/benchmarks/visualprocessbench/run_eval_sft_api.sh

# Resume but retry samples that previously hit API/runtime errors.
VPB_RETRY_ERRORS=1 bash VRPRM_v2.0/benchmarks/visualprocessbench/run_eval_sft_api.sh

# Resume but retry samples whose response could not be parsed as {"Score": [...]}.
VPB_RETRY_INVALID=1 bash VRPRM_v2.0/benchmarks/visualprocessbench/run_eval_sft_api.sh

# Override output paths and per-request timeout.
VPB_OUTPUT=/path/to/sft_stepwise_predictions.jsonl \
VPB_ERROR_LOG=/path/to/sft_stepwise_predictions.errors.jsonl \
VPB_STATUS_FILE=/path/to/sft_stepwise_predictions.status.json \
VPB_REQUEST_TIMEOUT=300 \
bash VRPRM_v2.0/benchmarks/visualprocessbench/run_eval_sft_api.sh
```

For stepwise mode, `VPB_WARMUP_MAX_TOKENS=1024` controls the first multimodal
chat request, and each step judgment only generates one token. In full mode,
`VPB_MAX_TOKENS=2048` controls the full `{"Score": [...]}` response. vLLM requires:

```text
input_tokens + max_tokens <= MAX_MODEL_LEN
```

If vLLM reports that `max_tokens` is too large, lower the eval-side generation budget:

```bash
VPB_MAX_TOKENS=1024 bash VRPRM_v2.0/benchmarks/visualprocessbench/run_eval_sft_api.sh
```

If GPU memory allows, you can instead restart vLLM with a larger context length:

```bash
MAX_MODEL_LEN=16384 bash VRPRM_v2.0/benchmarks/visualprocessbench/serve_sft_vllm_manual.sh /path/to/checkpoint-merged
```

Outputs:

```text
outputs/sft_stepwise_predictions.jsonl
outputs/sft_stepwise_predictions.errors.jsonl
outputs/sft_stepwise_predictions.status.json
outputs/sft_stepwise_predictions.metrics.json
```

## Evaluate Original VisualPRM

The local VisualPRM evaluator follows the official Hugging Face model card for `OpenGVLab/VisualPRM-8B`: it loads the model with `AutoTokenizer` and `AutoModel`, uses the official image preprocessing, and calls the remote-code method `generate_steps_with_soft_score()` for step-level scores.

For local Transformers evaluation:

```bash
export VISUALPRM_MODEL_PATH=/path/to/OpenGVLab/VisualPRM-8B
bash VRPRM_v2.0/benchmarks/visualprocessbench/run_eval_visualprm_paper.sh
```

## Compute Cost Benchmark

The compute-cost benchmark is split by deployment type.

### SFT vLLM Cost

For the paper setting, run the full VisualProcessBench once with global thinking and once with direct score generation. The script starts vLLM once and reuses the same service for both SFT modes:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_REPLICAS=4 \
SFT_MODEL_PATH=/path/to/merged-sft-model \
SFT_MODEL=auto \
SFT_CONCURRENCY=4 \
VPB_LIMIT=0 \
bash VRPRM_v2.0/benchmarks/visualprocessbench/run_benchmark_sft_vllm_paper_cost.sh
```

The two SFT modes are:

- `global_think_stepwise`: one global-thinking request plus one guided `0/1` request per step.
- `direct_scores`: no global thinking; one request per sample directly emits `{"Score": [...]}`.

For a smaller SFT smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 \
SFT_MODEL_PATH=/path/to/merged-sft-model \
SFT_START_VLLM=1 \
SFT_MODEL=auto \
VPB_LIMIT=128 \
bash VRPRM_v2.0/benchmarks/visualprocessbench/run_benchmark_sft_vllm_compute_cost.sh
```

If the SFT model is already served by vLLM, reuse it without starting a new server:

```bash
SFT_START_VLLM=0 \
SFT_BASE_URL=http://127.0.0.1:8000/v1 \
SFT_MODEL=vrprm-v2-sft \
VPB_LIMIT=128 \
bash VRPRM_v2.0/benchmarks/visualprocessbench/run_benchmark_sft_vllm_compute_cost.sh
```

For data-parallel vLLM replicas:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_REPLICAS=4 \
SFT_MODEL_PATH=/path/to/merged-sft-model \
SFT_CONCURRENCY=4 \
VPB_LIMIT=128 \
bash VRPRM_v2.0/benchmarks/visualprocessbench/run_benchmark_sft_vllm_compute_cost.sh
```

### VisualPRM Paper Cost

VisualPRM-8B keeps the same local Transformers path as `run_eval_visualprm_paper.sh`; the wrapper only measures wall time and summarizes the enriched prediction JSONL:

```bash
CUDA_VISIBLE_DEVICES=0 \
VISUALPRM_MODEL_PATH=/path/to/VisualPRM-8B \
VPB_LIMIT=0 \
bash VRPRM_v2.0/benchmarks/visualprocessbench/run_benchmark_visualprm_paper_compute_cost.sh
```

After all full runs finish, build a paper-ready table:

```bash
python VRPRM_v2.0/benchmarks/visualprocessbench/make_compute_cost_paper_table.py \
  --root VRPRM_v2.0/benchmarks/visualprocessbench/outputs/compute_cost \
  --min-samples 2866
```

Defaults:

- `SFT_MODE=global_think_stepwise`, matching the global-think plus per-step guided `0/1` scoring flow.
- `VPB_LIMIT=128` evaluates the first 128 samples by default, matching `run_eval_visualprm_paper.sh`. Set `VPB_LIMIT=0` for the full 2,866-sample benchmark.
- `SFT_START_VLLM=1` starts `serve_sft_vllm_dp.sh` when `SFT_MODEL_PATH` is provided. Set `SFT_START_VLLM=0` to reuse existing endpoints.
- `SFT_STOP_VLLM_AFTER=1` stops vLLM replicas when the SFT cost run exits.
- `SFT_BASE_URL` may be comma-separated for already-running data-parallel API replicas.

Outputs are written under `outputs/compute_cost/<run_name>/`:

- SFT: `summary.json`, `summary.csv`, `sft_<mode>_samples.jsonl/.csv`, and `sft_<mode>_requests.csv`.
- VisualPRM: `visualprm_paper_predictions.jsonl`, `visualprm_paper_summary.json/.csv`, and `visualprm_paper_samples.csv`.
- The reported fields include wall time, latency percentiles, throughput, request counts, token usage, generated tokens, errors, quality metrics, VisualPRM image preprocessing time, score time, text token counts, step-score counts, and peak CUDA memory.

The VisualPRM-8B remote-code model was written against an older Transformers generation API. If it fails in your current SFT environment with missing `generate`, `generation_config`, or tied-weight attributes, use a separate legacy environment:

```bash
conda create -n visualprm_legacy python=3.10 -y
conda activate visualprm_legacy
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r VRPRM_v2.0/benchmarks/visualprocessbench/requirements-visualprm-legacy.txt
```

The evaluator uses `VPB_THRESHOLD=0.85` by default because `generate_steps_with_soft_score()` returns a 0-1 soft score for each step. A threshold of `0.0` will mark almost every scored step as correct and makes the incorrect-step F1 collapse. The script also removes blank lines inside each step before joining steps with `\n\n`, matching VisualPRM's internal step splitter.

For a smoke test:

```bash
VISUALPRM_MODEL_PATH=/path/to/OpenGVLab/VisualPRM-8B \
VPB_LIMIT=10 \
bash VRPRM_v2.0/benchmarks/visualprocessbench/run_eval_visualprm_paper.sh
```

Outputs:

```text
outputs/visualprm_predictions.jsonl
outputs/visualprm_predictions.metrics.json
```

## Recompute Metrics

```bash
python VRPRM_v2.0/benchmarks/visualprocessbench/metrics.py \
  --predictions VRPRM_v2.0/benchmarks/visualprocessbench/outputs/sft_predictions.jsonl \
  --output VRPRM_v2.0/benchmarks/visualprocessbench/outputs/sft_predictions.metrics.json
```

The metrics file contains:

- `overall_step_macro_f1`: macro F1 across all non-neutral steps.
- `mean_source_macro_f1`: unweighted mean of per-source macro F1.
- `by_source`: per-source correct F1, incorrect F1, macro F1, sample count, evaluated step count.
