# tabfm-why — Beyond Leaderboards: A Mechanistic Comparison of Tabular Foundation Models

TabPFN ≈ TabICL > baselines — but the gap is concentrated in confusable classes, robustness, and calibration, not everywhere. This repo contains the full study: benchmark harness, 8 mechanistic probes, significance tests, plots, paper draft, and raw results.

## Key findings

- **Ranked:** TabPFN (0.872 acc) ≈ TabICL (0.868) > RF (0.841) > Linear/Linear > LightGBM > MLP; regression RMSE TabPFN 0.396 > TabICL 0.422 > LightGBM 0.496.
- **Honest stats:** TabPFN≈TabICL (p=0.23); FM-vs-RF not significant (p≈0.07) — the gap lives in one dataset (vehicle opel/saab: +27–37 pp recall).
- **Mechanisms:** genuine interaction modeling (XOR ≈0.99 vs 0.51 linear), best categorical exploitation, flattest missingness curves, best calibration (ECE ≈0.035 vs 0.13 LightGBM), minimal shortcut reliance, noise immunity.
- Full analysis: [`REPORT.md`](REPORT.md) · Paper draft: [`PAPER.md`](PAPER.md)

## Install

```bash
pip install tabpfn tabicl lightgbm scikit-learn pandas matplotlib scipy
```

TabPFN v2.5+ needs a free Prior Labs key (otherwise every `.fit()` opens a browser login tab):

```bash
# get a key at https://ux.priorlabs.ai/account (accept the license), then:
export TABPFN_TOKEN="<your-key>"   # Windows: $env:TABPFN_TOKEN="<your-key>"
```

## Reproduce

```bash
python benchmark.py                        # main benchmark -> results_main.csv
python benchmark_tabpfn_only.py <dataset>  # per-dataset TabPFN rerun (repeat per dataset)
python mechanistic.py part1                # interaction, cat/num, missingness, calibration
python mechanistic.py part3                # shortcut, subgroup (part2 = attribution)
python plots.py                            # 11 plots -> plot_*.png
python sigtest.py                          # paired tests -> results_significance.csv
```

CPU-only, ~30–60 min total. Protocol: 7 TabArena datasets, 80/20 stratified splits, seeds {0,1,2}, train ≤2000 rows.

## Repo layout

| File | What |
|---|---|
| `benchmark.py` / `benchmark_tabpfn_only.py` | Evaluation harness + TabPFN rerun |
| `mechanistic.py` | 8 mechanistic probes (P1–P8, see REPORT.md §3) |
| `plots.py` / `sigtest.py` | Figures / paired significance tests |
| `results_full.csv`, `results_mech_*.csv`, `results_significance.csv` | Raw results |
| `plot_*.png` | Figures |
| `REPORT.md` / `PAPER.md` | Full report / paper draft |

## Citation

```bibtex
@misc{tabfmwhy2026,
  title  = {Beyond Leaderboards: A Mechanistic Comparison of Tabular Foundation Models},
  year   = {2026},
  note   = {Code: https://github.com/ayuushmaan/tabfm-why}
}
```
