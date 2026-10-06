"""Single-set LimiX worker: fresh CUDA context per dataset."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = sys.argv[4] if len(sys.argv) > 4 else "29523"
sys.path.insert(0, "/marimo/limix")

import numpy as np, pandas as pd, torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from inference.predictor import LimiXPredictor

name, oid, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
OUT = "/marimo/limix_full.csv"
clf = LimiXPredictor(device=torch.device("cuda"),
                     model_path="/tmp/limix_cache/LimiX-2M-v101.ckpt",
                     inference_config="/marimo/limix/config/cls_default_noretrieval.json")
b = fetch_openml(data_id=oid, as_frame=True, parser="auto")
X = b.data.copy()
for c in X.columns:
    if X[c].dtype == bool:
        X[c] = X[c].astype(int)
cat = [c for c in X.columns if str(X[c].dtype) in ("object", "category", "string") or X[c].dtype == object]
num = [c for c in X.columns if c not in cat]
for c in num:
    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
for c in cat:
    X[c] = X[c].astype(str).fillna("MISSING")
if cat:
    X[cat] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(X[cat])
Xn = X.to_numpy(dtype=np.float32)
y = pd.Series(b.target).astype(str)
yn = y.map({c: i for i, c in enumerate(sorted(y.unique()))}).to_numpy(dtype=np.int64)
itr, ite = train_test_split(np.arange(len(Xn)), test_size=0.2, random_state=seed, stratify=yn)
rs = np.random.RandomState(seed)
if len(itr) > 2000:
    itr = rs.choice(itr, 2000, replace=False)
if len(ite) > 1000:
    ite = rs.choice(ite, 1000, replace=False)
pred = clf.predict(Xn[itr], yn[itr], Xn[ite], task_type="Classification")
score = accuracy_score(yn[ite], np.argmax(np.asarray(pred), axis=1))
pd.DataFrame([{"dataset": name, "model": "limix", "seed": seed, "metric": "acc",
               "score": score, "env": f"torch={torch.__version__} limix=patched-v101"}]).to_csv(
    OUT, mode="a", header=not os.path.exists(OUT) or os.path.getsize(OUT) == 0, index=False)
print(f"DONE {name}/{seed} acc={score:.4f}", flush=True)
