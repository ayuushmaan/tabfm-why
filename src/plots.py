"""Generate all report plots from CSVs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PILOT_R = RESULTS / "pilot"
PILOT_F = FIGURES / "pilot"
ORDER = ["TabPFN","TabICL","RandomForest","LightGBM","Linear","MLP"]
plt.rcParams.update({"figure.dpi":150,"font.size":9})

full = pd.read_csv(os.path.join(PILOT_R,"results_full.csv"))
cl = full[full.task=="class"]

# 1. ranked accuracy
m = cl.groupby("model")["accuracy"].mean().reindex(ORDER)
plt.figure(figsize=(7,3.5)); plt.bar(m.index, m.values)
plt.ylabel("mean accuracy (6 cls datasets x 3 seeds)"); plt.title("Ranked comparison: classification accuracy")
plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(os.path.join(PILOT_F,"plot_rank_accuracy.png")); plt.close()

# 2. dataset-wise heatmap
piv = cl.groupby(["dataset","model"])["accuracy"].mean().unstack().reindex(columns=ORDER)
fig, ax = plt.subplots(figsize=(8,3.5))
im = ax.imshow(piv.values, vmin=0.6, vmax=1.0)
ax.set_xticks(range(len(ORDER)), ORDER, rotation=20); ax.set_yticks(range(len(piv)), piv.index)
for i in range(len(piv)):
    for j in range(len(ORDER)):
        ax.text(j, i, f"{piv.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Dataset-wise mean accuracy"); fig.colorbar(im, ax=ax, label="accuracy")
plt.tight_layout(); plt.savefig(os.path.join(PILOT_F,"plot_dataset_heatmap.png")); plt.close()

# 3. regression panel
rg = full[full.dataset=="california"].groupby("model")[["rmse","r2"]].mean().reindex(ORDER)
fig, a = plt.subplots(1,2,figsize=(8,3.2))
a[0].bar(rg.index, rg["rmse"]); a[0].set_title("California housing RMSE (lower better)"); a[0].tick_params(axis="x",rotation=20)
a[1].bar(rg.index, rg["r2"]); a[1].set_title("California housing R2"); a[1].tick_params(axis="x",rotation=20)
plt.tight_layout(); plt.savefig(os.path.join(PILOT_F,"plot_regression.png")); plt.close()

# 4. interaction
inter = pd.read_csv(os.path.join(PILOT_R,"results_mech_interaction.csv"))
p = inter.pivot(index="model",columns="setting",values="acc").reindex(ORDER)
x=np.arange(len(ORDER)); w=0.35
plt.figure(figsize=(7,3.5)); plt.bar(x-w/2,p["linear"],w,label="linear"); plt.bar(x+w/2,p["xor"],w,label="XOR (needs interaction)")
plt.xticks(x,ORDER,rotation=15); plt.ylabel("accuracy"); plt.title("Feature-interaction probe: linear vs XOR synthetic"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(PILOT_F,"plot_interaction.png")); plt.close()

# 5. catnum deltas
cat = pd.read_csv(os.path.join(PILOT_R,"results_mech_catnum.csv"))
base = cat[cat.setting=="full"].set_index("model")["acc"]
for s in ["num-only","cat-only"]:
    d = (cat[cat.setting==s].set_index("model")["acc"]-base).reindex(ORDER)
    plt.figure(figsize=(7,3.2)); plt.bar(d.index,d.values); plt.axhline(0,color="k",lw=0.8)
    plt.ylabel("Δacc vs full"); plt.title(f"credit-g: {s} ablation"); plt.xticks(rotation=15); plt.tight_layout()
    plt.savefig(os.path.join(PILOT_F,f"plot_catnum_{s}.png")); plt.close()

# 6. missingness curves
mis = pd.read_csv(os.path.join(PILOT_R,"results_mech_missing.csv"))
mis["rate"]=mis.setting.str.extract(r"mcar(\d+)").astype(int)
plt.figure(figsize=(7,3.8))
for mm in ORDER:
    s=mis[mis.model==mm].sort_values("rate")
    plt.plot(s["rate"],s["acc"],marker="o",label=mm)
plt.xlabel("MCAR missing % (diabetes)"); plt.ylabel("accuracy"); plt.title("Missing-value robustness"); plt.legend(ncols=3,fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(PILOT_F,"plot_missing.png")); plt.close()

# 7. calibration: ECE + logloss bars
cb = pd.read_csv(os.path.join(PILOT_R,"results_mech_calib.csv")).set_index("model").reindex(ORDER)
fig,a=plt.subplots(1,2,figsize=(8,3.4))
a[0].bar(cb.index,cb["ece"]); a[0].set_title("ECE diabetes (lower better)"); a[0].tick_params(axis="x",rotation=20)
a[1].bar(cb.index,cb["logloss"]); a[1].set_title("Log-loss diabetes (lower better)"); a[1].tick_params(axis="x",rotation=20)
plt.tight_layout(); plt.savefig(os.path.join(PILOT_F,"plot_calibration.png")); plt.close()

# 8. shortcut drops
sc = pd.read_csv(os.path.join(PILOT_R,"results_mech_shortcut.csv")).set_index("model").reindex(ORDER)
plt.figure(figsize=(7,3.5)); plt.bar(sc.index,sc["drop_vs_clean"]); plt.axhline(0,color="k",lw=0.8)
plt.ylabel("Δacc with spurious col (test uncorr.)"); plt.title("Shortcut exploitation (smaller drop = less shortcut reliance)")
plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(os.path.join(PILOT_F,"plot_shortcut.png")); plt.close()

# 9. subgroup (vehicle)
sg = pd.read_csv(os.path.join(PILOT_R,"results_mech_subgroup.csv"))
p2 = sg.pivot(index="model",columns="setting",values="acc").reindex(ORDER)
x=np.arange(len(ORDER)); ww=0.18
plt.figure(figsize=(8,3.8))
for i,c in enumerate(p2.columns):
    plt.bar(x+(i-1.5)*ww, p2[c], ww, label=c.split(":")[1])
plt.xticks(x,ORDER,rotation=15); plt.ylabel("per-class recall"); plt.title("Vehicle subgroup recall (opel/saab are hard)"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(PILOT_F,"plot_subgroup.png")); plt.close()

# 10. label noise
ln = pd.read_csv(os.path.join(PILOT_R,"results_mech_labelnoise.csv"))
p3 = ln.pivot(index="model",columns="setting",values="acc").reindex(ORDER)
x=np.arange(len(ORDER))
plt.figure(figsize=(7,3.5)); plt.bar(x-0.2,p3["flip0"],0.4,label="clean"); plt.bar(x+0.2,p3["flip20"],0.4,label="20% label noise")
plt.xticks(x,ORDER,rotation=15); plt.ylabel("accuracy"); plt.title("Label-noise robustness (diabetes)"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(PILOT_F,"plot_labelnoise.png")); plt.close()

print("plots done:", [f for f in os.listdir(PILOT_F) if f.startswith("plot_")])
