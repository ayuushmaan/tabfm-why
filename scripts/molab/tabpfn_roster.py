"""TabPFN-3.5 roster (MOLAB). 16 sets x 3 seeds. Inference-only.
Caps 2000/1000, 80/20 stratified, ordinal-encoded NumPy. Resume-aware.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, mean_squared_error

OUT = "/marimo/tabpfn_roster.csv"
SETS = [
    ("credit-g", 31, "cls"),
    ("diabetes", 37, "cls"),
    ("breast-w", 15, "cls"),
    ("vehicle", 54, "cls"),
    ("segment", 36, "cls"),
    ("heart-statlog", 53, "cls"),
    ("balance-scale", 11, "cls"),
    ("tic-tac-toe", 50, "cls"),
    ("letter", 6, "cls"),
    ("mushroom", 24, "cls"),
    ("iris", 61, "cls"),
    ("car", 40978, "cls"),
    ("dresses-sales", 23381, "cls"),
    ("boston", 531, "reg"),
    ("cpu_act", 197, "reg"),
    ("puma8NH", 225, "reg"),
]
SEEDS = [0, 1, 2]


def load(name, oid, kind):
    try:
        b = fetch_openml(data_id=oid, as_frame=True, parser="auto")
        return b.data, b.target
    except Exception as e:
        print("SKIP", name, e, flush=True)
        return None, None


def prep(X, y, kind):
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    cat = [c for c in X.columns if str(X[c].dtype) in ("object", "category", "string") or X[c].dtype == object]
    num = [c for c in X.columns if c not in cat]
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        m = X[c].median() if X[c].notna().any() else 0
        X[c] = X[c].fillna(m)
    for c in cat:
        X[c] = X[c].astype(str).fillna("MISSING")
    if cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat] = enc.fit_transform(X[cat])
    Xn = X.to_numpy(dtype=np.float64)
    if kind == "cls":
        ys = pd.Series(y).astype(str)
        cls_list = sorted(ys.unique())
        mapd = {c: i for i, c in enumerate(cls_list)}
        ys = ys.map(mapd).to_numpy(dtype=np.int64)
        return Xn, ys
    else:
        ys = pd.to_numeric(pd.Series(y), errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        return Xn, ys


def main():
    import torch
    import sklearn
    import tabpfn

    if torch.cuda.is_available():
        devname = torch.cuda.get_device_name(0)
    else:
        devname = "cpu"
    env = f"torch={torch.__version__} numpy={np.__version__} pandas={pd.__version__} sklearn={sklearn.__version__} tabpfn={tabpfn.__version__} cuda={devname}"
    done = set()
    try:
        d = pd.read_csv(OUT)
        done = set(zip(d["dataset"], d["seed"]))
    except Exception:
        done = set()
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    for n, o, k in SETS:
        X, y = load(n, o, k)
        if X is None:
            continue
        X, y = prep(X, y, k)
        for s in SEEDS:
            if (n, s) in done:
                continue
            strat = y if k == "cls" else None
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s, stratify=strat)
            if len(Xtr) > 2000:
                Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=2000, random_state=s, stratify=(ytr if k == "cls" else None))
            if len(Xte) > 1000:
                Xte, _, yte, _ = train_test_split(Xte, yte, train_size=1000, random_state=s)
            t1 = time.time()
            try:
                if k == "cls":
                    m = TabPFNClassifier(device="cuda")
                    m.fit(Xtr, ytr)
                    pr = m.predict(Xte)
                    sc = float(accuracy_score(yte, pr))
                    ok = "OK"
                else:
                    m = TabPFNRegressor(device="cuda")
                    m.fit(Xtr, ytr)
                    pr = m.predict(Xte)
                    sc = float(- (mean_squared_error(yte, pr)) ** 0.5)
                    ok = "OK"
            except Exception as e:
                sc = float("nan")
                ok = "FAIL:" + str(e)[:230]
            t2 = time.time()
            row = dict(dataset=n, model="tabpfn", seed=s, score=sc, split="80_20_cap", n_train=len(Xtr), n_test=len(Xte), time_sec=round(t2 - t1, 2), version="tabpfn9.1.0", env=env)
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(n, s, ok, row["score"], flush=True)


if __name__ == "__main__":
    main()
