# SFT Data Generation Pipeline

This repository builds the **global-thinking + stepwise multi-turn SFT dataset** for a
visual reasoning process reward model from `VisualPRM400K-v1.1-Raw`.

The pipeline reads raw VisualPRM400K annotations, converts them to the VRPRM prompt
format, calls a multimodal OpenAI-compatible chat endpoint, and keeps only rollout
responses whose predicted process scores match the conservative Monte Carlo process
labels. A second-stage builder then merges accepted positive and negative rollouts into
one multi-turn conversation per source sample with a target step-level negative ratio.

## Scripts

```text
data_pipeline/rollout_teacher.py   # stage 1: global-thinking + step-level teacher rollout
data_pipeline/build_multiturn.py   # stage 2: multi-turn SFT dataset builder
data_pipeline/common.py            # shared library (paths, sampling, IO, prompts)
```

`rollout_visualprm_global_think_stepwise_sft_data_pipeline.py` matches a two-stage
inference flow: first generate one global `<think>...</think>` block, then score each
candidate step with a single `0/1` token. The script asks the rollout model for hidden
reasoning, compresses that hidden reasoning into global thinking with a second API call,
and expands each original problem into one optional `global_thinking` sample plus
multiple `step_score` samples.

## Input Data

Default input paths are relative to the repository root:

```text
data/VisualPRM400K-v1.1-Raw/annotations
data/VisualPRM400K-v1.1-Raw/images
```

Each raw annotation sample is expected to contain fields like:

```json
{
  "image": "VisualPRM400K-v1.1-Raw/Geometry3K/train/1471/img_diagram.png",
  "question": "<image>\n...",
  "answer": "52",
  "response": "...",
  "steps_with_score": [
    {
      "step": "...",
      "score": 0.125,
      "num_mc_correct": 2,
      "num_mc_total": 16
    }
  ]
}
```

## Requirements

```bash
pip install openai tqdm
```

## Stage 1 — Rollout

Run negative and positive subsets separately:

```bash
python data_pipeline/rollout_teacher.py \
  --polarity negative \
  --negative-score-threshold 0.125 \
  --positive-score-threshold 0.75 \
  --sample-total 10000 \
  --target-negative-step-ratio 0.4 \
  --num-workers 32

python data_pipeline/rollout_teacher.py \
  --polarity positive \
  --negative-score-threshold 0.125 \
  --positive-score-threshold 0.75 \
  --sample-total 10000 \
  --target-negative-step-ratio 0.4 \
  --num-workers 32
```

The first API call keeps hidden thinking enabled by default, asks the model to output
strict `{"Score": [...]}` step scores in visible content, and saves
`message.reasoning` / `message.reasoning_content` when the API returns it. A source
sample is kept only when the visible step scores match the conservative VisualPRM400K
MC labels on every confident step. The second call disables hidden thinking by default
and compresses the hidden reasoning into one visible `<think>...</think>` block.

Conservative step labels:

```text
negative: MC score <= 0.125
positive: MC score >= 0.75
ignored:  0.125 < MC score < 0.75
```

Ignored middle-score steps do not become `step_score` training rows. The compressed
global thinking is rejected if it is exactly `READY`, still contains `READY` after
cleanup, or is shorter than `--global-thinking-min-chars` (default `100`).

If your server requires an explicit thinking flag for the first call:

```bash
python data_pipeline/rollout_teacher.py \
  --polarity negative \
  --analysis-extra-body-json '{"chat_template_kwargs":{"thinking":true}}'
```

The saved rows have two task types:

```text
global_thinking: assistant is <think>compressed global thinking</think>
step_score:      assistant is exactly 1 or 0
```

`--include-global-thinking-task` is enabled by default so the same model can learn both
stages. Use `--no-include-global-thinking-task` for step-score rows only.
`--target-negative-step-ratio` optionally downsamples positive `step_score` rows after
rollout.

By default the scripts talk to a local OpenAI-compatible server
(`http://127.0.0.1:8000/v1`). Override with `--base-url` / `--api-key` or the
`OPENAI_BASE_URL` / `OPENAI_API_KEY` environment variables.

## Stage 2 — Build the Multi-Turn SFT Dataset

```bash
python data_pipeline/build_multiturn.py
```

Default inputs/output (all under `rollout_outputs/`):

```text
visualprm400k_negative_global_think_stepwise_sft_success.json   (input)
visualprm400k_positive_global_think_stepwise_sft_success.json   (input)
visualprm400k_global_think_stepwise_multiturn_sft_neg35.json    (output)
visualprm400k_global_think_stepwise_multiturn_sft_neg35.stats.json
```

The builder keeps all negative rollout samples, samples positive rollout samples until
the trainable step-level negative ratio is about `35%`, and creates a single `messages`
conversation per source sample:

```text
system
user: full question/reference/candidate solution with image
assistant: <think>global thinking / step overview</think>        loss_scale=0.5
user: current step 0 + previous judgments
assistant: 0 or 1                                               loss_scale=2.0
...
```

Middle-score uncertain steps are kept in the conversation history with `loss=false`, so
later step context stays continuous without training on low-confidence labels.

When training on this file, use assistant-level `loss_scale` support
(e.g. ms-swift `--loss_scale default --is_binary_loss_scale false`); the dataset uses
`loss_scale=0.5` for global-thinking turns and `loss_scale=2.0` for step-score turns.

## Published Data Products (Hugging Face Dataset)

The generated products are published as a Hugging Face Dataset (see the sibling
`huggingface_dataset/` folder or the dataset repository), not committed to this
code repository:

| File | Size | Content |
| --- | --- | --- |
| `visualprm400k_positive_global_think_stepwise_sft_success.json` | 58M | 12,571 accepted positive-polarity rollouts |
| `visualprm400k_positive_global_think_stepwise_sft_failed.json` | 15M | 1,237 rejected positive-polarity rollouts |
| `visualprm400k_negative_global_think_stepwise_sft_success.json` | 15M | 3,555 accepted negative-polarity rollouts |
| `visualprm400k_negative_global_think_stepwise_sft_failed.json` | 27M | 2,090 rejected negative-polarity rollouts |
| `visualprm400k_global_think_stepwise_multiturn_sft_neg35.json` | 9.1M | Final multi-turn SFT dataset |
| `visualprm400k_global_think_stepwise_multiturn_sft_neg35.stats.json` | 2K | Builder statistics |

Final dataset statistics (`neg35`):

```text
samples:                1378
trainable_steps:        5592
negative_trainable:     1954
positive_trainable:     3638
negative_ratio:         0.3494
ignored_steps:          1649
messages:               18616
global_loss_scale:      0.5
step_loss_scale:        2.0
```

Image paths inside the data are relative (`data/VisualPRM400K-v1.1-Raw/images/...`);
download the raw dataset so the paths resolve, as described in the dataset card.
Processing logs are runtime artifacts and are not published.

## Important Options

Shared options:

```text
--input-path           JSON/JSONL annotation file or annotation directory.
--image-root           Local VisualPRM400K images directory.
--output-path          Accepted SFT output JSON path.
--failed-output-path   Failed/rejected sample output JSON path.
--log-path             Per-sample rollout log path.
--model-name           Model name sent to the chat completion API.
--base-url             OpenAI-compatible API base URL.
--api-key              API key. Use EMPTY if your server ignores it.
--num-workers          Async rollout concurrency. Default is 32.
--max-retries          Number of retries after an API/generation exception.
--retry-sleep          Base sleep seconds between retries.
--sample-total         Total sampled rows across annotation files. Default is 5000. Use 0 to disable.
--samples-per-annotation
                       Optional override: sample up to this many rows per annotation file.
--seed                 Random seed for sampling.
--limit                Optional maximum number of samples.
--polarity             positive or negative.
--dry-run              Load, filter, sample, and print statistics without API calls.
```

Stage-1-specific options:

```text
--negative-score-threshold
                       Conservative negative step threshold. Default is 0.125.
--positive-score-threshold
                       Conservative positive step threshold. Default is 0.75.
--analysis-extra-body-json
                       Extra JSON for the first call. Default is empty so hidden reasoning is not disabled.
--compression-extra-body-json
                       Extra JSON for the compression call. Default disables hidden thinking.
--global-thinking-min-chars
                       Reject compressed global thinking shorter than this many chars. Default is 100.
--target-negative-step-ratio
                       Downsample positive step_score rows to this negative ratio after rollout.
--include-global-thinking-task / --no-include-global-thinking-task
                       Whether to keep global_thinking rows. Default: include.
```

Stage-2 builder options:

```text
--negative-path        Accepted negative rollout JSON. Default is the stage-1 negative output.
--positive-path        Accepted positive rollout JSON. Default is the stage-1 positive output.
--output-path          Output multi-turn dataset JSON.
--target-negative-ratio
                       Target trainable step-level negative ratio. Default is 0.35.
--seed                 Random seed for positive-sample selection.
```

## Rollout Acceptance Rule

A source sample is accepted only when the model's visible step scores match the
conservative binarized Monte Carlo labels on **every confident step**:

1. The response contains a valid `{"Score": [...]}` JSON object.
2. The number of predicted scores equals the number of original steps.
3. For every confident step, `(pred_score > 0) == (conservative_label > 0)`.

Steps with middle MC scores (`0.125 < score < 0.75`) are ignored for validation and
training. The original float scores are preserved in metadata.

## Validation Without Rollout

Syntax check:

```bash
python -m py_compile data_pipeline/rollout_teacher.py data_pipeline/build_multiturn.py data_pipeline/common.py
```

Argument check:

```bash
python data_pipeline/rollout_teacher.py --help
python data_pipeline/build_multiturn.py --help
```
