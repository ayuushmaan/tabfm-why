"""Error-pocket subgrouping (Box 1): vehicle, all 5 FMs + xgboost, seed 0.
For every misclassified test row: kNN distance to train, 5NN label purity,
predictive margin. Saves /marimo/error_pockets.csv
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29514"
sys.path.insert(0, "/marimo/tabdpt/src")
sys.path.insert(0, "/marimo/limix")

import numpy as np, pandas as pd, torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

OUT = "/marimo/error_pockets.csv"
SEED = 0

limix_clf = None

def predict(model, Xtr, ytr, Xte):
    if model == "tabpfn":
        from tabpfn import TabPFNClassifier
        m = TabPFNClassifier(device="cuda")
        m.fit(Xtr, ytr)
        return m.predict(Xte), m.predict_proba(Xte)
    if model == "tabicl":
        from tabicl import TabICLClassifier
        m = TabICLClassifier(device="cuda")
        m.fit(Xtr, ytr)
        return m.predict(Xte), m.predict_proba(Xte)
    if model == "tabdpt":
        from tabdpt import TabDPTClassifier
        m = TabDPTClassifier(device="cuda", compile=False, verbose=False)
        m.fit(Xtr.astype(np.float64), ytr.astype(np.int64))
        p = m.predict(Xte.astype(np.float64), seed=SEED)
        try:
            pr = m.predict_proba(Xte.astype(np.float64))
        except Exception:
            pr = None
        return p, pr
    if model == "limix":
        global limix_clf
        if limix_clf is None:
            from inference.predictor import LimiXPredictor
            limix_clf = LimiXPredictor(device=torch.device("cuda"),
                                       model_path="/tmp/limix_cache/LimiX-2M-v101.ckpt",
                                       inference_config="/marimo/limix/config/cls_default_noretrieval.json")
        pr = np.asarray(limix_clf.predict(Xtr.astype(np.float32), ytr, Xte.astype(np.float32),
                                          task_type="Classification"))
        return pr.argmax(1), pr
    if model == "mitra":
        from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier
        m = MitraClassifier()
        m.fit(Xtr.astype(np.float32), ytr)
        p = m.predict(Xte.astype(np.float32))
        try:
            pr = m.predict_proba(Xte.astype(np.float32))
        except Exception:
            pr = None
        return p, pr
    if model == "xgboost":
        from xgboost import XGBClassifier
        m = XGBClassifier(device="cuda", n_estimators=500, random_state=SEED, n_jobs=8)
        m.fit(Xtr, ytr)
        return m.predict(Xte), m.predict_proba(Xte)
    raise ValueError(model)

def main():
    print("loading vehicle", flush=True)
    b = fetch_openml(data_id=54, as_frame=True, parser="auto")
    X = b.data.copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
        if str(X[c].dtype) in ("object", "category", "string") or X[c].dtype == object:
            X[c] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(X[[c]].astype(str)).ravel()
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    Xn = X.to_numpy(dtype=np.float64)
    y = pd.Series(b.target).astype(str)
    classes = sorted(y.unique())
    yn = y.map({c: i for i, c in enumerate(classes)}).to_numpy(dtype=np.int64)
    itr, ite = train_test_split(np.arange(len(Xn)), test_size=0.2, random_state=SEED, stratify=yn)
    Xtr, ytr, Xte, yte = Xn[itr], yn[itr], Xn[ite], yn[ite]
    Xs = StandardScaler().fit_transform(np.vstack([Xtr, Xte]))
    nn = NearestNeighbors(n_neighbors=6).fit(StandardScaler().fit_transform(Xtr))
    dists, idxs = nn.kneighbors(StandardScaler().fit_transform(Xte))
    t0 = time.time()
    rows = []
    for model in ["tabpfn", "tabicl", "tabdpt", "limix", "mitra", "xgboost"]:
        try:
            pred, proba = predict(model, Xtr, ytr, Xte)
        except Exception as e:
            print(f"FAIL {model}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        pred = np.asarray(pred)
        err = np.where(pred != yte)[0]
        for j in test_idx_subset(err):
            d5, ix5 = dists[j, 1:], idxs[j, 1:]
            purity = float((ytr[ix5] == yte[j]).mean())
            if proba is not None:
                pr = np.asarray(proba)[j]
                margin = float(np.sort(pr)[-1] - np.sort(pr)[-2]) if len(pr) > 1 else 1.0
            else:
                margin = float("nan")
            rows.append({"model": model, "test_idx": int(j), "true": classes[yte[j]],
                         "pred": classes[pred[j]], "knn_dist": float(d5.mean()),
                         "purity": purity, "margin": margin})
        print(f"DONE {model}: {len(err)}/{len(yte)} errors [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"SAVED {len(rows)} error rows [{time.time()-t0:.0f}s]", flush=True)

def test_idx_subset(err):
    return err

if __name__ == "__main__":
    main()
