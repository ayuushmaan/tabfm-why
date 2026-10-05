"""TabDPT roster re-run (Box 1). Protocol: 80/20 stratified, seeds 0,1,2,
train cap 2000 / test cap 1000, ordinal-encoded, numpy+int inputs.
Resume-aware: appends to /marimo/tabdpt_roster.csv. Env pinned in header row.
"""
import os, sys, time, warnings
os.environ["TORCHDYNAMO_DISABLE"] = "1"
warnings.filterwarnings("ignore")
sys.path.insert(0, "/marimo/tabdpt/src")

import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, mean_squared_error

OUT = "/marimo/tabdpt_roster.csv"
SETS = [
    ("credit-g", 31, "cls"), ("diabetes", 37, "cls"), ("breast-w", 15, "cls"),
    ("vehicle", 54, "cls"), ("segment", 36, "cls"), ("heart-statlog", 53, "cls"),
    ("balance-scale", 11, "cls"), ("tic-tac-toe", 50, "cls"),
    ("letter", 6, "cls"), ("mushroom", 24, "cls"), ("iris", 61, "cls"),
    ("car", 40978, "cls"), ("dresses-sales", 23381, "cls"),
    ("boston", 455, "reg"), ("cpu_act", 197, "reg"), ("puma8NH", 225, "reg"),
]
SEEDS = [0, 1, 2]

def load(name, oid, kind):
    if kind == "reg" and name == "california":
        h = fetch_california_housing(as_frame=True)
        idx = h.data.sample(n=2500, random_state=0).index
        return h.data.loc[idx].reset_index(drop=True), h.target.loc[idx].reset_index(drop=True)
    try:
        b = fetch_openml(data_id=oid, as_frame=True, parser="auto")
    except Exception as e:
        print(f"SKIP {name}: {e}", flush=True)
        return None, None
    return b.data, b.target

def prep(X, y, kind, seed):
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    cat = [c for c in X.columns if str(X[c].dtype) in ("object", "category", "string") or X[c].dtype == object]
    num = [c for c in X.columns if c not in cat]
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median() if len(X[c].dropna()) else 0)
    for c in cat:
        X[c] = X[c].astype(str).fillna("MISSING")
    if cat:
        X[cat] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(X[cat])
    Xn = X.to_numpy(dtype=np.float64)
    if kind == "cls":
        y = pd.Series(y).astype(str)
        classes = sorted(y.unique())
        y = y.map({c: i for i, c in enumerate(classes)}).to_numpy(dtype=np.int64)
    else:
        y = pd.to_numeric(pd.Series(y), errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    return Xn, y

def main():
    import torch, sklearn, scipy
    env = f"torch={torch.__version__} numpy={np.__version__} pandas={pd.__version__} sklearn={sklearn.__version__} scipy={scipy.__version__} cuda={torch.cuda.get_device_name(0)}"
    print("ENV:", env, flush=True)
    done = set()
    try:
        d = pd.read_csv(OUT)
        done = set(zip(d["dataset"], d["seed"]))
    except FileNotFoundError:
        pass
    from tabdpt import TabDPTClassifier, TabDPTRegressor
    t0 = time.time()
    for name, oid, kind in SETS:
        X, y = load(name, oid, kind)
        if X is None:
            continue
        for seed in SEEDS:
            if (name, seed) in done:
                print(f"skip {name}/{seed} (done)", flush=True)
                continue
            Xn, yn = prep(X, y, kind, seed)
            strat = yn if kind == "cls" and len(np.unique(yn)) > 1 else None
            itr, ite = train_test_split(np.arange(len(Xn)), test_size=0.2, random_state=seed, stratify=strat)
            if len(itr) > 2000:
                itr = np.random.RandomState(seed).choice(itr, 2000, replace=False)
            if len(ite) > 1000:
                ite = np.random.RandomState(seed).choice(ite, 1000, replace=False)
            Xtr, ytr, Xte, yte = Xn[itr], yn[itr], Xn[ite], yn[ite]
            try:
                if kind == "cls":
                    m = TabDPTClassifier(device="cuda", compile=False, verbose=False)
                    m.fit(Xtr, ytr.astype(np.int64))
                    pred = m.predict(Xte, seed=seed)
                    score = accuracy_score(yte, pred)
                    metric = "acc"
                else:
                    m = TabDPTRegressor(device="cuda", compile=False, verbose=False)
                    m.fit(Xtr, ytr)
                    pred = m.predict(Xte)
                    score = float(np.sqrt(mean_squared_error(yte, pred)))
                    metric = "rmse"
            except Exception as e:
                print(f"FAIL {name}/{seed}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            row = pd.DataFrame([{"dataset": name, "kind": kind, "model": "tabdpt", "seed": seed,
                                 "metric": metric, "score": score, "env": env,
                                 "elapsed_s": round(time.time() - t0, 1)}])
            row.to_csv(OUT, mode="a", header=not os.path.exists(OUT) or os.path.getsize(OUT) == 0, index=False)
            print(f"DONE {name}/{seed} {metric}={score:.4f} [{time.time()-t0:.0f}s]", flush=True)
    print(f"ALL DONE [{time.time()-t0:.0f}s]", flush=True)

if __name__ == "__main__":
    main()
