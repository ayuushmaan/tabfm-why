"""Reproducible Tabular FM benchmark harness (CPU pilot).
Protocol mimics TabArena: stratified holdout, 3 seeds, subsampled to CPU-friendly sizes.
Models: TabPFN, TabICL, LightGBM, RandomForest, Linear, MLP.
Datasets: 7 TabArena OpenML members + California housing regression.
"""
import warnings, time, traceback
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    log_loss, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
import lightgbm as lgb

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SEEDS = [0, 1, 2]
MAX_TRAIN = 2000
MAX_TEST = 1000

def load_datasets():
    ds = {}
    # TabArena members via OpenML (small, CPU friendly)
    openml_ids = {
        "credit-g": 31,      # finance, binary, mixed (20 feats, 1000 rows)
        "diabetes": 37,      # medical, binary, numeric (768x8)
        "breast-w": 15,      # medical, binary, numeric (699x9)
        "vehicle": 54,       # 4-class numeric (846x18)
        "segment": 36,       # 7-class numeric vision (2310x19)
        "heart-statlog": 53, # medical binary (270x13, mixed)
    }
    for name, oid in openml_ids.items():
        try:
            print(f"Fetching {name} (OpenML {oid})...", flush=True)
            b = fetch_openml(data_id=oid, as_frame=True, parser="auto")
            X, y = b.data, b.target
            ds[name] = (X, y, "class")
        except Exception as e:
            print(f"  FAILED {name}: {e}")
    # Regression: California housing subsampled (domain: geo/econ)
    try:
        print("Fetching California housing...", flush=True)
        h = fetch_california_housing(as_frame=True)
        idx = h.data.sample(n=2500, random_state=0).index
        X, y = h.data.loc[idx], h.target.loc[idx]
        ds["california"] = (X.reset_index(drop=True), y.reset_index(drop=True), "reg")
    except Exception as e:
        print(f"  FAILED california: {e}")
    return ds

def prep_frame(X):
    X = X.copy()
    # coerce object/category; convert bools
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    cat_cols = [c for c in X.columns if str(X[c].dtype) in ("object","category","string") or X[c].dtype==object]
    num_cols = [c for c in X.columns if c not in cat_cols]
    # force numeric coercion where possible
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, cat_cols, num_cols

def make_preprocess(cat_cols, num_cols, for_tree=False):
    # Trees handle ordinal + missing natively-ish; linear/MLP need impute+scale/one-hot-lite (ordinal+scale)
    if for_tree:
        trans = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                               ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat_cols),
        ]) if cat_cols and num_cols else "passthrough"
        return trans
    trans = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                           ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                           ("sc", StandardScaler())]), cat_cols),
    ]) if cat_cols and num_cols else "passthrough"
    # handle single-type frames
    if not cat_cols:
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    if not num_cols:
        return Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                         ("sc", StandardScaler())])
    return trans

def ece_score(y_true, y_prob, n_bins=10):
    # multiclass: confidence = max prob
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:
        conf = y_prob; pred = (y_prob > 0.5).astype(int)
    else:
        conf = y_prob.max(axis=1); pred = y_prob.argmax(axis=1)
        # map y_true to ints if strings
        if y_true.dtype == object:
            classes = np.unique(y_true); mp = {c:i for i,c in enumerate(classes)}
            y_true = np.array([mp[v] for v in y_true])
    acc = (pred == y_true)
    edges = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i+1])
        if m.sum() > 0:
            ece += np.abs(acc[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)

def get_models(task, n_classes=2):
    models = {}
    if task == "class":
        models["LightGBM"] = lgb.LGBMClassifier(verbosity=-1, n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, random_state=0)
        models["RandomForest"] = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0)
        models["Linear"] = LogisticRegression(max_iter=2000, C=1.0)
        models["MLP"] = MLPClassifier(hidden_layer_sizes=(128,64), early_stopping=True, n_iter_no_change=10, max_iter=500, random_state=0)
        try:
            from tabpfn import TabPFNClassifier
            models["TabPFN"] = TabPFNClassifier(n_estimators=4, random_state=0)
        except Exception as e:
            print("TabPFN unavailable:", e)
        try:
            from tabicl import TabICLClassifier
            models["TabICL"] = TabICLClassifier(random_state=0)
        except Exception as e:
            print("TabICL unavailable:", e)
    else:
        models["LightGBM"] = lgb.LGBMRegressor(verbosity=-1, n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, random_state=0)
        models["RandomForest"] = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0)
        models["Linear"] = Ridge(alpha=1.0)
        models["MLP"] = MLPRegressor(hidden_layer_sizes=(128,64), early_stopping=True, n_iter_no_change=10, max_iter=500, random_state=0)
        try:
            from tabpfn import TabPFNRegressor
            models["TabPFN"] = TabPFNRegressor(n_estimators=4, random_state=0)
        except Exception as e:
            print("TabPFNReg unavailable:", e)
        try:
            from tabicl import TabICLRegressor
            models["TabICL"] = TabICLRegressor(random_state=0)
        except Exception as e:
            print("TabICLReg unavailable:", e)
    return models

def run():
    ds = load_datasets()
    rows = []
    for dname, (Xraw, yraw, kind) in ds.items():
        Xraw = Xraw.reset_index(drop=True)
        y = pd.Series(yraw).reset_index(drop=True)
        if kind == "class":
            # normalize target to string labels then encode per-split
            y = y.astype(str)
        X, cat_cols, num_cols = prep_frame(Xraw)
        print(f"\n=== {dname}: shape={X.shape}, cats={len(cat_cols)}, nums={len(num_cols)}, task={kind} ===", flush=True)
        task = "reg" if kind == "reg" else "class"
        for seed in SEEDS:
            strat = y if task=="class" else None
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)
            if len(Xtr) > MAX_TRAIN:
                Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=MAX_TRAIN, random_state=seed, stratify=(ytr if task=="class" else None))
            if len(Xte) > MAX_TEST:
                _, Xte, _, yte = train_test_split(Xte, yte, test_size=MAX_TEST, random_state=seed, stratify=(yte if task=="class" else None))
            # label encode for metrics that need ints
            from sklearn.preprocessing import LabelEncoder
            le = None
            if task=="class":
                le = LabelEncoder().fit(ytr)
                ytr_e, yte_e = le.transform(ytr), le.transform(yte)
                n_classes = len(le.classes_)
            models = get_models(task, n_classes if task=="class" else 0)
            for mname, model in models.items():
                try:
                    t0 = time.time()
                    Xt_tr, Xt_te = Xtr.copy(), Xte.copy()
                    if mname in ("TabPFN","TabICL"):
                        for c in cat_cols:
                            Xt_tr[c] = Xt_tr[c].astype("category")
                            Xt_te[c] = Xt_te[c].astype("category")
                        # simple median/mode impute for numerics (TabPFN handles NaN but ensure numeric dtype)
                        for c in num_cols:
                            med = Xt_tr[c].median()
                            Xt_tr[c] = Xt_tr[c].fillna(med); Xt_te[c] = Xt_te[c].fillna(med)
                        model.fit(Xt_tr, ytr if task=="class" else ytr.astype(float))
                        if task=="class":
                            pred = model.predict(Xt_te)
                            proba = model.predict_proba(Xt_te) if hasattr(model,"predict_proba") else None
                        else:
                            pred = model.predict(Xt_te); proba=None
                    else:
                        prep = make_preprocess(cat_cols, num_cols, for_tree=(mname in ("LightGBM","RandomForest")))
                        pipe = Pipeline([("prep", prep), ("m", model)])
                        pipe.fit(Xtr, ytr_e if task=="class" else ytr.astype(float))
                        if task=="class":
                            pred_e = pipe.predict(Xte)
                            pred = le.inverse_transform(pred_e)
                            proba = pipe.predict_proba(Xte) if hasattr(pipe,"predict_proba") else None
                        else:
                            pred = pipe.predict(Xte); proba=None
                    dt = time.time()-t0
                    rec = {"dataset":dname,"task":task,"seed":seed,"model":mname,"time_s":round(dt,2),
                           "n_train":len(Xtr),"n_test":len(Xte),"n_cat":len(cat_cols),"n_num":len(num_cols)}
                    if task=="class":
                        rec["accuracy"]=accuracy_score(yte, pred)
                        rec["bal_acc"]=balanced_accuracy_score(yte_e, le.transform(pred))
                        rec["f1_macro"]=f1_score(yte_e, le.transform(pred), average="macro", zero_division=0)
                        if proba is not None:
                            try: rec["logloss"]=log_loss(yte_e, proba, labels=list(range(n_classes)))
                            except: rec["logloss"]=np.nan
                            try:
                                if n_classes==2: rec["auc"]=roc_auc_score(yte_e, proba[:,1])
                                else: rec["auc"]=roc_auc_score(yte_e, proba, multi_class="ovr")
                            except: rec["auc"]=np.nan
                            try: rec["ece"]=ece_score(yte_e, proba)
                            except: rec["ece"]=np.nan
                        else:
                            rec["logloss"]=np.nan; rec["auc"]=np.nan; rec["ece"]=np.nan
                    else:
                        rec["rmse"]=float(np.sqrt(mean_squared_error(yte.astype(float), pred)))
                        rec["mae"]=float(mean_absolute_error(yte.astype(float), pred))
                        rec["r2"]=float(r2_score(yte.astype(float), pred))
                    rows.append(rec)
                    print(f"  seed{seed} {mname}: {dt:.1f}s " + (f"acc={rec.get('accuracy',0):.3f} ll={rec.get('logloss',float('nan')):.3f}" if task=="class" else f"rmse={rec.get('rmse',0):.3f} r2={rec.get('r2',0):.3f}"), flush=True)
                except Exception as e:
                    print(f"  seed{seed} {mname} FAILED: {e}")
                    traceback.print_exc()
                    rows.append({"dataset":dname,"task":task,"seed":seed,"model":mname,"error":str(e)[:200]})
    out = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "results_main.csv", index=False)
    print("\nSaved results_main.csv with", len(out), "rows")
    print(out.groupby(["model"]).mean(numeric_only=True).round(4))

if __name__ == "__main__":
    run()
