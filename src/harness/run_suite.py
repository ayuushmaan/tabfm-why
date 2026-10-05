"""Resume-aware suite runner. One shard = one CLI call (keep each <4h on MOLAB 6h cap).
Example: python -m harness.run_suite --models Linear,RandomForest --datasets heart-statlog --seeds 0
"""
import argparse, json, sys, time, traceback, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error

from .core import Manifest, make_run_id, STORE
from .adapters import build

def ece_score(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    conf = y_prob.max(axis=1); pred = y_prob.argmax(axis=1)
    acc = (pred == y_true)
    edges = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() > 0:
            ece += np.abs(acc[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)

def prep_frame(X):
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    cat = [c for c in X.columns if str(X[c].dtype) in ("object", "category", "string") or X[c].dtype == object]
    num = [c for c in X.columns if c not in cat]
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, cat, num

def load_dataset(name):
    from sklearn.datasets import fetch_openml
    oid = {"credit-g": 31, "diabetes": 37, "breast-w": 15, "vehicle": 54,
           "segment": 36, "heart-statlog": 53}[name]
    b = fetch_openml(data_id=oid, as_frame=True, parser="auto")
    return b.data, pd.Series(b.target).astype(str).reset_index(drop=True), "class"

def run_one(manifest, model_name, dataset, seed, max_train=2000, max_test=1000):
    config = {"max_train": max_train, "max_test": max_test}
    rid = make_run_id(model_name, dataset, seed, config)
    if manifest.status(rid) == "done":
        print(f"skip {rid} (done)"); return
    t0 = time.time()
    try:
        Xraw, y, task = load_dataset(dataset)
        Xraw = Xraw.reset_index(drop=True)
        X, cat_cols, num_cols = prep_frame(Xraw)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        if len(Xtr) > max_train:
            Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=max_train, random_state=seed, stratify=ytr)
        if len(Xte) > max_test:
            _, Xte, _, yte = train_test_split(Xte, yte, test_size=max_test, random_state=seed, stratify=yte)
        row_ids = Xte.index.values
        le = LabelEncoder().fit(ytr)
        ytr_e, yte_e = le.transform(ytr), le.transform(yte)
        m = build(model_name, "class", seed)
        if model_name in ("TabPFN", "TabICL"):
            a, b = Xtr.copy(), Xte.copy()
            for c in cat_cols:
                a[c] = a[c].astype("category"); b[c] = b[c].astype("category")
            for c in num_cols:
                med = a[c].median(); a[c] = a[c].fillna(med); b[c] = b[c].fillna(med)
            t1 = time.time(); m.fit(a, ytr); fit_s = time.time() - t1
            t1 = time.time(); pred = m.predict(b); proba = m.predict_proba(b); pred_s = time.time() - t1
            pred_e = le.transform(pd.Series(pred).astype(str))
        else:
            treeish = model_name in ("LightGBM", "RandomForest", "XGBoost", "CatBoost")
            if cat_cols and num_cols:
                num_t = SimpleImputer(strategy="median") if treeish else Pipeline(
                    [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
                cat_t = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                  ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
                if not treeish:
                    cat_t.steps.append(("sc", StandardScaler()))
                prep = ColumnTransformer([("num", num_t, num_cols), ("cat", cat_t, cat_cols)])
            elif cat_cols:
                prep = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                 ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
            else:
                prep = SimpleImputer(strategy="median") if treeish else Pipeline(
                    [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
            pipe = Pipeline([("prep", prep), ("m", m)])
            yt = ytr if model_name != "MLP" else pd.Series(le.transform(ytr), index=ytr.index)
            t1 = time.time(); pipe.fit(Xtr, yt); fit_s = time.time() - t1
            t1 = time.time()
            raw = pipe.predict(Xte)
            if model_name == "MLP":
                pred_e = np.asarray(raw, dtype=int)
                pred = le.inverse_transform(pred_e)
            else:
                pred = np.asarray(raw).astype(str)
                pred_e = le.transform(pd.Series(pred).astype(str))
            proba = pipe.predict_proba(Xte); pred_s = time.time() - t1
        pred_df = pd.DataFrame({
            "row_id": row_ids,
            "y_true": np.asarray(yte), "y_pred": np.asarray(pred).astype(str),
            "y_proba": [json.dumps([round(float(v), 6) for v in row]) for row in proba],
        })
        metrics = {"accuracy": float(accuracy_score(yte_e, pred_e)),
                   "logloss": float(log_loss(yte_e, proba, labels=list(range(len(le.classes_))))),
                   "ece": float(ece_score(yte_e, np.asarray(proba))),
                   "fit_s": round(fit_s, 2), "predict_s": round(pred_s, 2),
                   "wall_s": round(time.time() - t0, 2)}
        meta = {"model": model_name, "dataset": dataset, "seed": seed,
                "config": json.dumps(config), "n_train": len(Xtr), "n_test": len(Xte)}
        manifest.store_run(rid, meta, pred_df, metrics)
        print(f"done {rid} acc={metrics['accuracy']:.3f} wall={metrics['wall_s']}s", flush=True)
    except Exception as e:
        traceback.print_exc()
        manifest.mark(rid, "failed", error=str(e)[:300])
        print(f"FAILED {rid}: {e}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Linear,RandomForest")
    ap.add_argument("--datasets", default="heart-statlog")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--max-train", type=int, default=2000)
    ap.add_argument("--max-test", type=int, default=1000)
    a = ap.parse_args()
    man = Manifest()
    for ds in a.datasets.split(","):
        for seed in map(int, a.seeds.split(",")):
            for mn in a.models.split(","):
                run_one(man, mn.strip(), ds.strip(), seed, a.max_train, a.max_test)
    print("store at", STORE)

if __name__ == "__main__":
    main()
