# SFT Environment Configuration List

Recommended runtime:

- Python: 3.10 or 3.11
- PyTorch: 2.8.0 + CUDA 12.8 wheel
- GPU: default script uses one H200 via `CUDA_VISIBLE_DEVICES=0`
- Precision: bf16
- Framework: local `VRPRM_v2.0/ms-swift`
- Base model: `Qwen/Qwen3-VL-8B-Instruct (Hub id or local snapshot)`
- Dataset: `VRPRM_v2.0/rollout_outputs/rollout_sft_success_0614_0615.json`

Install checklist:

```bash
conda activate vrprm_sft

# Remove the current cu130 PyTorch stack first.
pip uninstall -y torch torchvision torchaudio flash-attn flash_attn

# Install PyTorch 2.8.0 with CUDA 12.8 wheels.
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Verify the CUDA runtime and ABI before choosing the flash-attn wheel.
python - <<'PY'
import torch, sys
print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cxx11 abi:", torch._C._GLIBCXX_USE_CXX11_ABI)
PY

cd <path-to>/VRPRM_v2.0/sft
pip install -r requirements-sft.txt
```

Install the prebuilt flash-attn wheel that matches Python 3.10, torch 2.8, Linux x86_64, and the reported ABI.

If `cxx11 abi: True`:

```bash
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
```

If `cxx11 abi: False`:

```bash
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
```

Final verification:

```bash
python - <<'PY'
import torch, flash_attn
print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)
print(flash_attn.__version__)
PY
```

If `flash-attn` is unavailable on the machine, update `sft/env.list`:

```bash
export ATTN_IMPL="sdpa"
export PACKING="false"
export PADDING_FREE="false"
```

Main environment variables are listed in `env.list`. The most commonly changed ones are:

- `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`: GPU layout.
- `OUTPUT_DIR`: checkpoint output directory.
- `TUNER_TYPE`: default `lora_llm`, meaning LLM LoRA plus full ViT/aligner training.
- `MAX_LENGTH`: default `8192`, set lower if memory is tight.
- `IMAGE_MAX_TOKEN_NUM`: default `1024`, set lower if image tokens cause OOM.
- `GRADIENT_ACCUMULATION_STEPS`: controls global batch size.

For a single H200, the default config uses `PER_DEVICE_TRAIN_BATCH_SIZE=2` and
`GRADIENT_ACCUMULATION_STEPS=4`. If memory is still underutilized, try:

```bash
export PER_DEVICE_TRAIN_BATCH_SIZE="4"
export PER_DEVICE_EVAL_BATCH_SIZE="4"
```

Memory-saving fallback:

```bash
export TUNER_TYPE="lora"
export VIT_LR=""
export ALIGNER_LR=""
export VIT_GRADIENT_CHECKPOINTING="false"
export MAX_LENGTH="4096"
export IMAGE_MAX_TOKEN_NUM="768"
```

The train script skips `--vit_lr` and `--aligner_lr` when these variables are empty.
