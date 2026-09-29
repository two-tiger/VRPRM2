# VLMEval Bo128 Evaluation from Cached Rollouts


> **Note:** rollout caches (`outputs*/`) and downloaded VLMEval TSVs
> (`datasets/`) are runtime artifacts and are **not** committed. Generate them
> with the scripts below (`download_table2_datasets.sh`,
> `run_generate_rollouts_cache.sh`, ...).

This folder is organized around cached `ROLLOUT_N=128` policy rollouts under
`outputs/`. Existing rollout caches are resumable: candidate-level records are
stored as soon as they finish, and sample-level records are written after all
128 candidates for that sample are available.

Current workflow:

1. Generate or resume missing Bo128 policy rollout caches.
2. Evaluate Bo1 policy ability directly from cached rollouts.
3. Start a PRM OpenAI-compatible endpoint (`run_bon_vrprm.sh` fans out over
   `PRM_BASE_URL` endpoints).
4. Score cached `rollouts_n128.jsonl` files with the PRM.
5. Select BoN predictions from cached PRM scores.
6. Run VLMEvalKit evaluation and summarize results.

## Cached Rollouts

The retained cache files are:

```text
outputs/<MODEL_LABEL>/<DATASET>/rollouts_n128.jsonl
outputs/<MODEL_LABEL>/<DATASET>/rollouts_n128.candidates.jsonl
```

`rollouts_n128.jsonl` is the sample-level aggregate used by PRM scoring.
`rollouts_n128.candidates.jsonl` stores candidate-level rollout records and is
kept as the resumable generation cache.

Current cached model/dataset coverage:

```text
InternVL2.5-8B:  MMMU, MathVista, MathVision, MathVerse-VO, WeMath, LogicVista
InternVL2.5-26B: MMMU, MathVista, MathVision, MathVerse-VO, WeMath, LogicVista
InternVL2.5-38B: MMMU, MathVista, MathVision, MathVerse-VO(partial)
```

> **Action required (paper Table 4 integrity):** the committed cache state for
> `InternVL2.5-38B` was incomplete when last checked (`MathVerse-VO` 661/788,
> `WeMath`/`LogicVista` missing). The paper reports 38B on all six benchmarks.
> Before the numbers are frozen, regenerate/complete the 38B caches and
> re-derive Table 4 with `analysis/make_paper_table4_bon.py` so every reported
> cell traces to a complete cache. (Historical note: 661/788 was the last
> observed state.) The candidate cache already contains
partial progress for 787 indices, so rerunning generation will only request
missing candidates. `WeMath` and `LogicVista` are not present for 38B yet.

`DynaMath` is part of the default benchmark list, but no DynaMath Bo128 rollout
cache is currently present under `outputs/`. Add or generate:

```text
outputs/<MODEL_LABEL>/DynaMath/rollouts_n128.jsonl
outputs/<MODEL_LABEL>/DynaMath/rollouts_n128.candidates.jsonl
```

before running the default all-dataset commands. To evaluate only currently
cached datasets, override `DATASETS` without `DynaMath`.

## Environment

Use a separate evaluation environment from vLLM/SGLang serving environments.
Python 3.10 is recommended.

```bash
conda create -n vlmeval_bon python=3.10 -y
conda activate vlmeval_bon

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r VRPRM_v2.0/benchmarks/vlmeval_bon/requirements-vlmeval-bon.txt
```

The local TSV files are already under:

```text
VRPRM_v2.0/benchmarks/vlmeval_bon/datasets
```

`common.py` sets `LMUData` to this local directory by default. If you want to
use another VLMEvalKit data directory:

```bash
export LMUData=/path/to/VLMEvalData
```

## Stage -1: Generate or Resume Bo128 Rollouts

Start an OpenAI-compatible policy endpoint first. For `InternVL2.5-38B` with
vLLM on four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH=OpenGVLab/InternVL2_5-38B \
SERVED_MODEL_NAME=internvl25-38b \
PORT=8000 \
TENSOR_PARALLEL_SIZE=4 \
MAX_MODEL_LEN=8192 \
MAX_NUM_SEQS=32 \
bash VRPRM_v2.0/benchmarks/vlmeval_bon/serve_internvl25_38b_vllm.sh
```

Use a local model directory in `MODEL_PATH` if the checkpoint is already
downloaded. The script starts a background vLLM server, waits for
`/v1/models`, and prints the `POLICY_BASE_URL`/`POLICY_MODEL` values for
generation. Stop it with:

```bash
bash VRPRM_v2.0/benchmarks/vlmeval_bon/serve_internvl25_38b_vllm.sh stop
```

If vLLM fails during CUDA graph or custom all-reduce initialization, use the
transformers fallback server. For faster rollout generation, start four
data-parallel single-GPU replicas:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH=OpenGVLab/InternVL2_5-38B \
SERVED_MODEL_NAME=internvl25-38b \
PORT_BASE=8000 \
NUM_REPLICAS=4 \
DEVICE_MAP=split \
DTYPE=bf16 \
POLICY_CONCURRENCY=4 \
POLICY_REQUEST_TIMEOUT=1800 \
bash VRPRM_v2.0/benchmarks/vlmeval_bon/serve_internvl25_38b_transformers_dp4.sh
```

The script starts ports `8000`-`8003` and prints a comma-separated
`POLICY_BASE_URL`. `generate_rollouts.py` will round-robin requests across
these endpoints. Each replica serializes its own model inference internally, so
set `POLICY_CONCURRENCY` around the number of replicas first. Stop replicas
with:

```bash
bash VRPRM_v2.0/benchmarks/vlmeval_bon/serve_internvl25_38b_transformers_dp4.sh stop
```

Then run the cache generator.
It skips complete samples, reuses successful candidates in
`rollouts_n128.candidates.jsonl`, retries failed/missing candidates, and writes
`rollouts_n128.jsonl` once a sample has all 128 candidates.

For the current `InternVL2.5-38B` gap:

```bash
MODEL_LABEL=InternVL2.5-38B \
POLICY_BASE_URL=http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1 \
POLICY_MODEL=internvl25-38b \
DATASETS=MathVerse-VO,WeMath,LogicVista \
ROLLOUT_N=128 \
POLICY_CONCURRENCY=4 \
POLICY_ROLLOUT_CONCURRENCY=1 \
bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_generate_rollouts_cache.sh
```

Useful knobs:

```bash
POLICY_MAX_TOKENS=2048
POLICY_TEMPERATURE=0.7
POLICY_TOP_P=0.95
POLICY_MAX_RETRIES=3
POLICY_REQUEST_TIMEOUT=600
NO_PROGRESS=1
```

Use `OVERWRITE=1` only when intentionally rebuilding a rollout cache from
scratch.

## Stage 0: Evaluate Bo1 Policy Ability

Bo1 evaluates the policy model itself without PRM reranking. It directly selects
candidate `0` from each cached `rollouts_n128.jsonl` record and writes a
VLMEvalKit-compatible prediction file.

```bash
MODEL_LABEL=InternVL2.5-8B \
DATASETS=MMMU,MathVista,MathVision,MathVerse-VO,DynaMath,WeMath,LogicVista \
ROLLOUT_N=128 \
PREDICTION_MODE=raw \
bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_eval_bo1_from_rollouts.sh
```

Outputs:

```text
outputs/<MODEL_LABEL>/<DATASET>/bon1_policy_from_n128_selected.xlsx
outputs/<MODEL_LABEL>/<DATASET>/bon1_policy_from_n128_selection.json
outputs/<MODEL_LABEL>/<DATASET>/bon1_policy_from_n128_eval.json
```

`PREDICTION_MODE=raw` keeps the exact first rollout response and is the default
for matching the cached model output. If you need a stricter final-answer-only
sanity check:

```bash
PREDICTION_MODE=final bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_eval_bo1_from_rollouts.sh
```

If candidate `0` has an empty or failed rollout and you want to keep evaluating
that sample using the first valid candidate:

```bash
FALLBACK_FIRST_VALID=1 bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_eval_bo1_from_rollouts.sh
```

Use `RUN_EVAL=0` to only write selected prediction `.xlsx` files and skip
VLMEvalKit evaluation.

## Start PRM Server

Option A: four single-GPU vLLM replicas. This is usually efficient for many
short stepwise scoring requests.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH=/path/to/prm-or-merged-sft-model \
SERVED_MODEL_NAME=vrprm-sft \
PORT_BASE=8001 \
MAX_NUM_SEQS=32 \
PRM_CONCURRENCY=64 \
bash VRPRM_v2.0/benchmarks/vlmeval_bon/serve_prm_vllm_4gpu.sh
```

The script starts ports `8001`-`8004` and prints the `PRM_BASE_URL` value to use
for scoring. Stop the replicas with:

```bash
bash VRPRM_v2.0/benchmarks/vlmeval_bon/serve_prm_vllm_4gpu.sh stop
```

Option B: SGLang data-parallel PRM serving.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH=/path/to/prm-or-merged-sft-model \
SERVED_MODEL_NAME=vrprm-sft \
PORT=8001 \
TP=1 \
DP=4 \
MAX_RUNNING_REQUESTS=128 \
MAX_QUEUED_REQUESTS=1024 \
PRM_CONCURRENCY=128 \
bash VRPRM_v2.0/benchmarks/vlmeval_bon/serve_prm_sglang_dp4.sh
```

## Stage 1: Score Cached Rollouts with PRM

Run PRM scoring from the retained Bo128 rollout cache:

```bash
MODEL_LABEL=InternVL2.5-8B \
DATASETS=MMMU,MathVista,MathVision,MathVerse-VO,DynaMath,WeMath,LogicVista \
PRM_BASE_URL=http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1 \
PRM_MODEL=vrprm-sft \
ROLLOUT_N=128 \
PRM_CONCURRENCY=64 \
bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_score_rollouts_cache.sh
```

Outputs:

```text
outputs/<MODEL_LABEL>/<DATASET>/rollouts_n128_prm_scores.jsonl
outputs/<MODEL_LABEL>/<DATASET>/rollouts_n128_prm_scores.candidates.jsonl
```

Scoring is resumable by candidate. Rerun the same command after interruption;
existing successful candidate scores are skipped. Use `OVERWRITE=1` only when
you want to rebuild scores from scratch.

The default reward mode is `stepwise`, matching the VisualProcessBench-style
global thinking plus per-step 0/1 scoring path:

1. Generate one concise `<think>...</think>` context for the candidate solution.
2. Query each step with `max_tokens=1` and optional guided choice `["1", "0"]`.

For SGLang, use:

```bash
PRM_GUIDED_CHOICE=0 bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_score_rollouts_cache.sh
```

## Stage 2: Select and Evaluate PRM BoN

Select from cached PRM scores and run VLMEvalKit evaluation:

```bash
MODEL_LABEL=InternVL2.5-8B \
DATASETS=MMMU,MathVista,MathVision,MathVerse-VO,DynaMath,WeMath,LogicVista \
ROLLOUT_N=128 \
BON_LIST=1,2,4,8,16,32,64,128 \
bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_eval_bon_from_cache.sh
```

Use `RUN_EVAL=0` to only write selected prediction `.xlsx` files and skip the
judge/evaluator calls:

```bash
RUN_EVAL=0 bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_eval_bon_from_cache.sh
```

Use `ALLOW_PARTIAL=1` only for debugging incomplete PRM score caches. Keep it
off for final numbers.

Stage-2 outputs:

```text
outputs/<MODEL_LABEL>/<DATASET>/bon<N>_prm_from_n128_selected.xlsx
outputs/<MODEL_LABEL>/<DATASET>/bon<N>_prm_from_n128_selection.json
outputs/<MODEL_LABEL>/<DATASET>/bon<N>_prm_from_n128_eval.json
```

## VPB-Style Qwen3 SFT PRM BoN

To evaluate the SFT PRM checkpoint with the same global-thinking plus per-step
chat scoring logic used by
`benchmarks/visualprocessbench/api_eval_global_think_stepwise.py`, first serve
the merged checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH=VRPRM_v2.0/sft/output/qwen3_vl_8b_thinking_global_stepwise_multiturn_sft/v0-20260628-015250/checkpoint-513-merge \
SERVED_MODEL_NAME=qwen3-vl-8b-thinking-global-stepwise-sft \
PORT_BASE=8000 \
NUM_REPLICAS=4 \
MAX_MODEL_LEN=16384 \
MAX_NUM_SEQS=16 \
ENFORCE_EAGER=1 \
PRM_CONCURRENCY=32 \
bash VRPRM_v2.0/benchmarks/visualprocessbench/serve_sft_vllm_dp.sh "${MODEL_PATH}"
```

This starts four single-GPU vLLM replicas on ports `8000`-`8003`, which is
usually faster than one tensor-parallel endpoint for many independent PRM
scoring requests. `ENFORCE_EAGER=1` disables vLLM torch.compile/cudagraph
startup, avoiding corrupted compile-cache failures during multi-replica launch.
Set `ENFORCE_EAGER=0` only after clearing the vLLM torch compile cache and
confirming all replicas start reliably.

For a two-GPU machine, run the same script with `CUDA_VISIBLE_DEVICES=0,1`; the
replica count, default ports, log directory, and recommended scorer concurrency
are inferred automatically.

Then score cached policy rollouts, select the highest-scoring answer from the
first N rollouts for each BoN setting, and run VLMEvalKit evaluation. The
default scorer uses the token-logprob probability of the `1` judgment as each
step score and selects candidates by the PRM score only. Final-answer format and
correctness are not checked during PRM selection; the selected responses are
passed to VLMEvalKit for evaluation. All generated score caches, selection files,
and eval files are written to a separate output folder:

```bash
MODEL_LABELS=InternVL2.5-38B \
DATASETS=MMMU,MathVista,MathVision,MathVerse-VO,WeMath,LogicVista \
INPUT_ROOT=VRPRM_v2.0/benchmarks/vlmeval_bon/outputs \
OUTPUT_ROOT=VRPRM_v2.0/benchmarks/vlmeval_bon/outputs_qwen3_vl_8b_thinking_global_stepwise_bon \
PRM_BASE_URL=http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1 \
PRM_MODEL=qwen3-vl-8b-thinking-global-stepwise-sft \
ROLLOUT_N=128 \
BON_LIST=2,4,8,16,32,64,128 \
PRM_CONCURRENCY=32 \
PRM_THINK_MAX_TOKENS=1024 \
PRM_USE_LOGPROB_SCORE=1 \
bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_eval_qwen3_sft_global_stepwise_bon.sh
```

The scorer is resumable. For each dataset it writes:

```text
outputs_qwen3_vl_8b_thinking_global_stepwise_bon/<MODEL_LABEL>/<DATASET>/rollouts_n128_qwen3_sft_global_stepwise_logprob_final_answer_scores.jsonl
outputs_qwen3_vl_8b_thinking_global_stepwise_bon/<MODEL_LABEL>/<DATASET>/rollouts_n128_qwen3_sft_global_stepwise_logprob_final_answer_scores.candidates.jsonl
outputs_qwen3_vl_8b_thinking_global_stepwise_bon/<MODEL_LABEL>/<DATASET>/bon<N>_qwen3_sft_global_stepwise_logprob_final_answer_from_n128_selected.xlsx
outputs_qwen3_vl_8b_thinking_global_stepwise_bon/<MODEL_LABEL>/<DATASET>/bon<N>_qwen3_sft_global_stepwise_logprob_final_answer_from_n128_selection.json
outputs_qwen3_vl_8b_thinking_global_stepwise_bon/<MODEL_LABEL>/<DATASET>/bon<N>_qwen3_sft_global_stepwise_logprob_final_answer_from_n128_eval.json
```

Use `RUN_EVAL=0` to only create selected `.xlsx` files, or `RUN_SCORE=0` to
rerun selection/evaluation from an existing score cache. Use `OVERWRITE=1` when
you want to discard an existing score cache for the same reward label.

## VisualPRM BoN

VisualPRM BoN uses the same local `OpenGVLab/VisualPRM-8B`
`generate_steps_with_soft_score()` logic as
`benchmarks/visualprocessbench/run_eval_visualprm_paper.sh`. Run scoring and
selection on the GPU server with the VisualPRM checkpoint:

```bash
VISUALPRM_MODEL_PATH=/path/to/OpenGVLab/VisualPRM-8B \
MODEL_LABELS=InternVL2.5-8B \
DATASETS=MMMU,MathVista,MathVision,MathVerse-VO,WeMath,LogicVista \
INPUT_ROOT=VRPRM_v2.0/benchmarks/vlmeval_bon/outputs \
OUTPUT_ROOT=VRPRM_v2.0/benchmarks/vlmeval_bon/outputs_visualprm_bon \
ROLLOUT_N=128 \
BON_LIST=2,4,8,16,32,64,128 \
bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_visualprm_bon_select.sh
```

This writes VisualPRM score caches and selected prediction `.xlsx` files, but
does not call VLMEvalKit evaluation. After copying the selected outputs to a
server with network access, run:

```bash
MODEL_LABELS=InternVL2.5-8B \
DATASETS=MMMU,MathVista,MathVision,MathVerse-VO,WeMath,LogicVista \
OUTPUT_ROOT=VRPRM_v2.0/benchmarks/vlmeval_bon/outputs_visualprm_bon \
ROLLOUT_N=128 \
BON_LIST=2,4,8,16,32,64,128 \
EVAL_API_NPROC=4 \
bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_eval_visualprm_bon_vlmeval.sh
```

## Pass@K And Major@K

There are two Pass@K/Major@K modes.

The fast local script below uses exact/numeric/choice matching directly against
the cached rollout answers. It is useful as a cheap sanity check, but it is not
the same metric as VLMEvalKit for datasets with custom official aggregation.
For example, WeMath `Score (Strict)` groups step questions by knowledge unit and
normalizes over 525 units, so local `pass@1` row accuracy is expected to differ
from the Bo1 `Score (Strict)`.

```bash
MODEL_LABELS=InternVL2.5-8B \
DATASETS=MMMU,MathVista,MathVision,MathVerse-VO,WeMath,LogicVista \
INPUT_ROOT=VRPRM_v2.0/benchmarks/vlmeval_bon/outputs \
OUTPUT_ROOT=VRPRM_v2.0/benchmarks/vlmeval_bon/outputs_pass_major \
ROLLOUT_N=128 \
K_LIST=1,2,4,8,16,32,64,128 \
bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_eval_pass_major_from_rollouts.sh
```

To align with the same VLMEvalKit scoring used by Bo1/BoN, use the official
script. On the rollout server, first generate the prediction files only:

```bash
MODEL_LABELS=InternVL2.5-8B \
DATASETS=MMMU,MathVista,MathVision,MathVerse-VO,WeMath,LogicVista \
INPUT_ROOT=VRPRM_v2.0/benchmarks/vlmeval_bon/outputs \
OUTPUT_ROOT=VRPRM_v2.0/benchmarks/vlmeval_bon/outputs_pass_major_vlmeval \
ROLLOUT_N=128 \
K_LIST=1,2,4,8,16,32,64,128 \
BUILD_PREDICTIONS=1 \
RUN_EVAL=0 \
AGGREGATE=0 \
bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_eval_pass_major_vlmeval.sh
```

After copying `outputs_pass_major_vlmeval` to a server with judge/API access,
run VLMEvalKit evaluation and aggregate official scores:

```bash
MODEL_LABELS=InternVL2.5-8B \
DATASETS=MMMU,MathVista,MathVision,MathVerse-VO,WeMath,LogicVista \
OUTPUT_ROOT=VRPRM_v2.0/benchmarks/vlmeval_bon/outputs_pass_major_vlmeval \
ROLLOUT_N=128 \
K_LIST=1,2,4,8,16,32,64,128 \
BUILD_PREDICTIONS=0 \
RUN_EVAL=1 \
AGGREGATE=1 \
EVAL_API_BASE=http://your-api-host/v1/chat/completions \
EVAL_API_KEY=sk-your-key \
EVAL_API_NPROC=4 \
bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_eval_pass_major_vlmeval.sh
```

For WeMath, `WEMATH_PRIMARY=strict` is the default, so `pass@1` uses the same
`Score (Strict)` formula as Bo1.

If the rollout cache is already on a server with judge/API access, no GPU model
deployment is needed for this stage. You can build prediction files, run
VLMEvalKit, and aggregate the final Pass@K/Major@K table in one command. This
uses the optimized path by default: Pass@K only evaluates samples that have not
passed yet, and Major@K uses local final-answer extraction plus majority vote.

```bash
MODEL_LABELS=InternVL2.5-8B \
DATASETS=MMMU,MathVista,MathVision,MathVerse-VO,WeMath,LogicVista \
INPUT_ROOT=VRPRM_v2.0/benchmarks/vlmeval_bon/outputs \
OUTPUT_ROOT=VRPRM_v2.0/benchmarks/vlmeval_bon/outputs_pass_major_vlmeval \
ROLLOUT_N=128 \
K_LIST=1,2,4,8,16,32,64,128 \
BUILD_PREDICTIONS=1 \
RUN_EVAL=1 \
AGGREGATE=1 \
EVAL_API_NPROC=4 \
bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_eval_pass_major_vlmeval.sh
```

Set `OPTIMIZED=0` only when you intentionally want the older full evaluation
that evaluates every candidate file and every Major@K prediction file.

## Summarize

Collect per-dataset results into one CSV:

Bo1 policy summary:

```bash
python VRPRM_v2.0/benchmarks/vlmeval_bon/summarize_table2.py \
  --root VRPRM_v2.0/benchmarks/vlmeval_bon/outputs \
  --models InternVL2.5-8B,InternVL2.5-26B,InternVL2.5-38B \
  --datasets MMMU,MathVista,MathVision,MathVerse-VO,DynaMath,WeMath,LogicVista \
  --bon 1 \
  --rollout-n 128 \
  --reward-label policy \
  --output VRPRM_v2.0/benchmarks/vlmeval_bon/outputs/bo1_policy_summary.csv
```

PRM BoN summary:

```bash
python VRPRM_v2.0/benchmarks/vlmeval_bon/summarize_table2.py \
  --root VRPRM_v2.0/benchmarks/vlmeval_bon/outputs \
  --models InternVL2.5-8B,InternVL2.5-26B,InternVL2.5-38B \
  --bon 128 \
  --rollout-n 128 \
  --reward-label prm \
  --output VRPRM_v2.0/benchmarks/vlmeval_bon/outputs/bon128_summary.csv
```

## Useful Variables

```bash
MODEL_LABEL=InternVL2.5-8B
DATASETS=MMMU,MathVista,MathVision,MathVerse-VO,DynaMath,WeMath,LogicVista
OUTPUT_ROOT=VRPRM_v2.0/benchmarks/vlmeval_bon/outputs
ROLLOUT_N=128
BON_LIST=1,2,4,8,16,32,64,128
CANDIDATE_IDX=0
PREDICTION_MODE=raw
FALLBACK_FIRST_VALID=0
PRM_CONCURRENCY=64
PRM_REWARD_MODE=stepwise
PRM_WARMUP_MAX_TOKENS=1024
PRM_REQUEST_TIMEOUT=300
RUN_EVAL=1
EVAL_API_NPROC=4
EVAL_JUDGE=chatgpt-0125
NO_PROGRESS=1
```

## Remaining Scripts

- `common.py`: shared dataset, image, prompt, and JSON helpers.
- `download_table2_datasets.sh`: optional TSV downloader for another machine.
- `generate_rollouts.py`: resumable BoN policy rollout generator.
- `run_generate_rollouts_cache.sh`: stage -1 rollout generation runner.
- `select_bo1_from_rollouts.py`: Bo1 policy selection directly from rollout cache.
- `run_eval_bo1_from_rollouts.sh`: Bo1 policy evaluation runner.
- `score_rollouts_with_prm.py`: PRM scoring over cached policy rollouts.
- `score_rollouts_global_think_stepwise.py`: VPB-style global-thinking chat stepwise scorer.
- `select_bon_from_scores.py`: BoN selection from PRM score caches.
- `evaluate_vlmeval.py`: VLMEvalKit evaluator wrapper.
- `summarize_table2.py`: result CSV summarizer.
- `run_score_rollouts_cache.sh`: stage-1 scoring runner.
- `run_eval_bon_from_cache.sh`: stage-2 BoN selection/evaluation runner.
- `run_eval_qwen3_sft_global_stepwise_bon.sh`: Qwen3 SFT PRM BoN score/select/eval runner.
- `serve_internvl25_38b_vllm.sh`: vLLM policy server helper for InternVL2.5-38B.
- `serve_internvl25_38b_transformers.py`: OpenAI-compatible transformers policy server.
- `serve_internvl25_38b_transformers.sh`: launcher for the transformers policy server.
- `serve_internvl25_38b_transformers_dp4.sh`: four-replica transformers policy server helper.
- `../visualprocessbench/serve_sft_vllm_dp.sh`: vLLM SFT PRM server helper, defaulting to four data-parallel replicas.
- `serve_prm_vllm_4gpu.sh`: four-replica vLLM PRM server helper.
- `serve_prm_sglang_dp4.sh`: SGLang data-parallel PRM server helper.
