# VRPRM SFT (ms-swift)

| Script | Purpose |
| --- | --- |
| `train_vrprm_sft.sh` | **Paper SFT**: Qwen3-VL-8B-Thinking + LoRA(rank16/alpha32) on the 1,378-conversation multiturn dataset; think loss 0.5 / step loss 2.0 / uncertain loss=false; 3 epochs, lr 1e-4. Auto-validates the dataset (accepts both think and no-think variants). |
| `train_vrprm_sft_nothink.sh` | M5 ablation: same 5,592 step labels, think turn removed. Run `python data_pipeline/build_multiturn.py --no-think` first. |
| `train_vrprm_sft_40k_noncot.sh` | M3/M2 control: SFT from base on the same 40K non-CoT examples as RL. Run `data_pipeline/build_noncot_sft_from_rl.py` first. |
| `merge_lora.sh` | Merge LoRA -> HF model for vLLM (run-dir checkpoint auto-selection, tokenizer fix). |

Environment: see `env.list`, `environment.md`, `requirements-sft.txt`. All knobs are
environment variables inside each script. Distributed training via `USE_TORCHRUN=true`.

Historical variants (Instruct-base v1, explicit-think, single-turn thinking) are on
branch `archive/snapshot-202609`.
