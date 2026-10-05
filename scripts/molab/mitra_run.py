"""Mitra-v2 5-way (Box 2). Direct MitraClassifier, NumPy inputs, int labels.
Ordinal-encode categoricals first (pandas3/AutoGluon friction). Fine-tuning
~1-2 min/fit. Resume-aware CSV.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier

OUT = "/marimo/mitra_5way.csv"
SETS = [("diabetes", 37), ("credit-g", 31), ("vehicle", 54),
        ("breast-w", 15), ("segment", 36)]
SEEDS = [0, 1, 2]

def main():
    import torch, sklearn
    env = f"torch={torch.__version__} sklearn={sklearn.__version__} pandas={pd.__version__}"
    print("ENV:", env, flush=True)
    done = set()
    try:
        d = pd.read_csv(OUT)
        done = set(zip(d["dataset"], d["seed"]))
    except FileNotFoundError:
        pass
    t0 = time.time()
    for name, oid in SETS:
        try:
            b = fetch_openml(data_id=oid, as_frame=True, parser="auto")
        except Exception as e:
            print(f"SKIP {name}: {e}", flush=True)
            continue
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
        for seed in SEEDS:
            if (name, seed) in done:
                print(f"skip {name}/{seed}", flush=True)
                continue
            itr, ite = train_test_split(np.arange(len(Xn)), test_size=0.2, random_state=seed, stratify=yn)
            rs = np.random.RandomState(seed)
            if len(itr) > 2000:
                itr = rs.choice(itr, 2000, replace=False)
            if len(ite) > 1000:
                ite = rs.choice(ite, 1000, replace=False)
            try:
                m = MitraClassifier()
                m.fit(Xn[itr], yn[itr])
                score = accuracy_score(yn[ite], m.predict(Xn[ite]))
            except Exception as e:
                print(f"FAIL {name}/{seed}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            pd.DataFrame([{"dataset": name, "model": "mitra", "seed": seed,
                           "metric": "acc", "score": score, "env": env}]).to_csv(
                OUT, mode="a", header=not os.path.exists(OUT) or os.path.getsize(OUT) == 0, index=False)
            print(f"DONE {name}/{seed} acc={score:.4f} [{time.time()-t0:.0f}s]", flush=True)
    print(f"ALL DONE [{time.time()-t0:.0f}s]", flush=True)

if __name__ == "__main__":
    main()
