# VPB Compute-Cost Probes (paper Table 3)

Latency / generated-token / per-step overhead measurement. API evaluation
(vLLM-served VRPRM) and VisualPRM local evaluation (transformers) need
separate environments — run the two sides separately and join them on the
same selected ids.

| Script | What it measures |
| --- | --- |
| `run_api_backend_cost.sh` | Both backends (API PRM + VisualPRM) on a sampled subset via `benchmark_compute_cost.py`. |
| `run_sft_vllm_cost.sh` | SFT checkpoint served by vLLM: serves, evaluates, summarizes. |
| `run_sft_paper_cost.sh` | Paper-config variant of the SFT API cost run. |
| `run_vrprm_think_cost.sh` | VRPRM thinking-mode cost on a subset (API environment). |
| `run_visualprm_subset_cost.sh` | VisualPRM on the same selected ids (VisualPRM environment); join with the API-side run via `VPB_SELECTED_IDS_FILE`. |
| `run_visualprm_paper_cost.sh` | Full VisualPRM paper-protocol cost run + `summarize_compute_cost.py`. |
| `run_think_ablation_cost.sh` | Combined think on/off ablation cost analysis. |

Outputs land under `outputs/compute_cost/<run-name>/`; generate the paper
table with `make_compute_cost_paper_table.py` (analysis/ holds the paper-table
copy aligned with Table 3).
