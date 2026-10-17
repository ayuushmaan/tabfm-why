"""RQ2 follow-up on the finance/health domain sets.

Joins the stored 7-model domain AUCs (results/domain/domain_*.csv) with the
landmarker locality proxies (results/domain/landmarkers.csv) and tests whether
the FM advantage tracks WEAK local structure, as in the main RQ2 (docs/RQ2.md).

Writes:
  results/domain/rq2_domain_corr.csv   Spearman rho/p for each locality proxy
  figures/domain/rq2_locality.png      scatter FM-advantage vs 1NN-AUC / smoothness
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = r"C:\Users\ayush\OneDrive\Desktop\Theatre\Guido\Projects\TFM4.0"
RD = os.path.join(ROOT, "results", "domain")
FIG = os.path.join(ROOT, "figures", "domain")

ORDER = ["tabpfn", "tabicl", "tabdpt", "mitra", "limix", "ag", "xgb"]
DISP = {"tabpfn": "TabPFN", "tabicl": "TabICL", "tabdpt": "TabDPT", "mitra": "Mitra",
        "limix": "LimiX", "ag": "AutoGluon", "xgb": "XGBoost"}
FM = ["tabpfn", "tabicl", "tabdpt", "mitra", "limix"]
BASE = ["xgb", "ag"]
LOC = ["lm_1nn_auc", "lm_1nn_acc", "smooth_macro", "smooth_micro",
       "lm_tree_acc", "lm_logreg_acc", "tree_minus_linear"]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    n = len(a)
    if n < 4:
        return float("nan"), float("nan"), n
    try:
        from scipy.stats import spearmanr
        r = spearmanr(a, b)
        return float(r.statistic), float(r.pvalue), n
    except Exception:
        ra = pd.Series(a).rank().to_numpy()
        rb = pd.Series(b).rank().to_numpy()
        rho = np.corrcoef(ra, rb)[0, 1]
        t = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
        from math import erfc, sqrt
        p = erfc(abs(t) / sqrt(2))
        return float(rho), float(p), n


dm = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RD, "domain_*.csv"))])
auc = dm[dm["metric"] == "auc"].pivot_table(index="dataset", columns="model",
                                            values="score", aggfunc="mean")
dom = dm.drop_duplicates("dataset").set_index("dataset")["domain"].to_dict()
auc["fm"] = auc[FM].mean(axis=1)
auc["base"] = auc[BASE].mean(axis=1)
auc["adv_base"] = auc["fm"] - auc["base"]
auc["adv_xgb"] = auc["fm"] - auc["xgb"]
auc["adv_ag"] = auc["fm"] - auc["ag"]

lm = pd.read_csv(os.path.join(RD, "landmarkers.csv"))
lmw = lm.pivot_table(index="dataset", columns="metric", values="score", aggfunc="mean")

df = auc.join(lmw, how="inner")
df = df.dropna(subset=["adv_base"])
print("joined datasets:", len(df), "| finance:", (df.index.map(dom) == "finance").sum(),
      "health:", (df.index.map(dom) == "health").sum())

rows = []
for ycol in ["adv_base", "adv_xgb", "adv_ag"]:
    for xcol in LOC:
        rho, p, n = spearman(df[xcol], df[ycol])
        rows.append(dict(y=ycol, x=xcol, rho=rho, p=p, n=n, exclude="none"))
    nod = df[df.index != "creditcard-fraud"]
    for xcol in LOC:
        rho, p, n = spearman(nod[xcol], nod[ycol])
        rows.append(dict(y=ycol, x=xcol, rho=rho, p=p, n=n, exclude="creditcard-fraud"))
corr = pd.DataFrame(rows)
corr.to_csv(os.path.join(RD, "rq2_domain_corr.csv"), index=False)

print("\n=== Spearman(FM advantage, locality) — adv_base = mean(FM) - mean(XGB,AG) ===")
show = corr[(corr.y == "adv_base") & (corr.exclude == "none")].sort_values("rho")
print(show[["x", "rho", "p", "n"]].to_string(index=False, float_format=lambda v: "%.4f" % v))

print("\n=== sensitivity: excluding creditcard-fraud (adv_base) ===")
show2 = corr[(corr.y == "adv_base") & (corr.exclude == "creditcard-fraud")].sort_values("rho")
print(show2[["x", "rho", "p", "n"]].to_string(index=False, float_format=lambda v: "%.4f" % v))

print("\n=== per-dataset FM advantage vs locality (adv_base) ===")
tab = pd.DataFrame({
    "domain": df.index.map(dom),
    "adv_base": df["adv_base"],
    "adv_xgb": df["adv_xgb"],
    "lm_1nn_auc": df["lm_1nn_auc"],
    "smooth_macro": df["smooth_macro"],
}).sort_values("adv_base", ascending=False)
print(tab.to_string(float_format=lambda v: "%.4f" % v))

# ---- figure ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
colmap = {"finance": "#1f77b4", "health": "#d62728"}
for ax, xcol, title in [(axes[0], "lm_1nn_auc", "1NN AUC (local structure)"),
                        (axes[1], "smooth_macro", "neighborhood smoothness (macro)")]:
    for dname in ["finance", "health"]:
        sub = df[df.index.map(dom) == dname]
        ax.scatter(sub[xcol], sub["adv_base"], s=42, alpha=0.85,
                   color=colmap[dname], label=dname, edgecolor="white", linewidth=0.6)
    rho, p, n = spearman(df[xcol], df["adv_base"])
    ax.set_xlabel(title)
    ax.set_ylabel("FM advantage (mean FM AUC - mean baseline AUC)")
    ax.set_title("%s\nSpearman rho = %.2f (p = %.3f, n = %d)" % (title, rho, p, n), fontsize=10)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
fig.suptitle("RQ2 on finance/health: FM advantage tracks weak local structure", fontsize=12)
fig.tight_layout()
os.makedirs(FIG, exist_ok=True)
fig.savefig(os.path.join(FIG, "rq2_locality.png"), dpi=150)
print("\nwrote", os.path.join(FIG, "rq2_locality.png"))
