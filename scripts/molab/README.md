# MOLAB GPU run scripts (October 2026 session, 2× RTX PRO 6000)

Exact scripts executed on the boxes, preserved for reproducibility.
Common protocol: 80/20 stratified splits, seeds {0,1,2}, train cap 2000 /
test cap 1000, per-row env pinning, resume-aware CSVs.

| Script | Box | Output |
|---|---|---|
| `tabdpt_run.py` | 1 | `results/roster/tabdpt_roster.csv` (48 rows) |
| `xgb_run.py` | 2 | `results/xgboost/xgb_roster.csv` (48 rows) |
| `boston_fix.py` | both | corrected OpenML 455 → 531 |
| `mitra_run.py` | 2 | `results/roster/mitra_5way.csv` (15 rows) |
| `limix_patch.py` | 1 | ckpt config translation (weights untouched) |
| `limix_run.py` | 1 | `results/roster/limix_5way.csv` (15 rows) |
| `dilution_run.py` | 1 | `results/probes/dilution.csv` (60 rows) |
| `pockets_run.py` | 1 | `results/probes/error_pockets.csv` (152 rows) |
| `ag_run.py` | 2 | `results/ceiling/ag_ceiling.csv` (48 rows) |

Secrets (`TABPFN_TOKEN`, `HF_TOKEN`) never committed; bake into the kernel
env before launching token-gated jobs.
