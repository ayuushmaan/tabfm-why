"""TabPFN-only rerun with identical protocol to benchmark.py. Headless-safe (browser blocked)."""
import warnings, time, traceback, sys, os
warnings.filterwarnings("ignore")
import webbrowser
webbrowser.open = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("browser blocked headless"))
import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    log_loss, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score)

PY = r"C:\Users\ayush\OneDrive\Desktop\Theatre\Guido\Projects\TFM4.0"
SEEDS = [0, 1, 2]
MAX_TRAIN, MAX_TEST = 2000, 1000

def ece_score(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    conf = y_prob.max(axis=1); pred = y_prob.argmax(axis=1)
    acc = (pred == y_true)
    edges = np.linspace(0, 1, n_bins+1); ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i+1])
        if m.sum() > 0:
            ece += np.abs(acc[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)

def load_one(dname):
    if dname == "california":
        h = fetch_california_housing(as_frame=True)
        idx = h.data.sample(n=2500, random_state=0).index
        return h.data.loc[idx].reset_index(drop=True), h.target.loc[idx].reset_index(drop=True), "reg"
    oid = {"credit-g":31,"diabetes":37,"breast-w":15,"vehicle":54,"segment":36,"heart-statlog":53}[dname]
    b = fetch_openml(data_id=oid, as_frame=True, parser="auto")
    return b.data, b.target, "class"

def prep(X):
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == bool: X[c] = X[c].astype(int)
    cat = [c for c in X.columns if str(X[c].dtype) in ("object","category","string") or X[c].dtype==object]
    num = [c for c in X.columns if c not in cat]
    for c in num: X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, cat, num

def run_dataset(dname):
    Xraw, yraw, kind = load_one(dname)
    Xraw = Xraw.reset_index(drop=True); y = pd.Series(yraw).reset_index(drop=True)
    if kind == "class": y = y.astype(str)
    X, cat_cols, num_cols = prep(Xraw)
    task = "reg" if kind == "reg" else "class"
    rows = []
    for seed in SEEDS:
        strat = y if task=="class" else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)
        if len(Xtr) > MAX_TRAIN:
            Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=MAX_TRAIN, random_state=seed, stratify=(ytr if task=="class" else None))
        if len(Xte) > MAX_TEST:
            _, Xte, _, yte = train_test_split(Xte, yte, test_size=MAX_TEST, random_state=seed, stratify=(yte if task=="class" else None))
        try:
            t0=time.time()
            Xt_tr, Xt_te = Xtr.copy(), Xte.copy()
            for c in cat_cols:
                Xt_tr[c]=Xt_tr[c].astype("category"); Xt_te[c]=Xt_te[c].astype("category")
            for c in num_cols:
                med=Xt_tr[c].median(); Xt_tr[c]=Xt_tr[c].fillna(med); Xt_te[c]=Xt_te[c].fillna(med)
            if task=="class":
                from tabpfn import TabPFNClassifier
                le=LabelEncoder().fit(ytr); ytr_e,yte_e=le.transform(ytr),le.transform(yte)
                m=TabPFNClassifier(n_estimators=4, random_state=0)
                m.fit(Xt_tr, ytr); pred=m.predict(Xt_te); proba=m.predict_proba(Xt_te)
                ncl=len(le.classes_); dt=time.time()-t0
                rec={"dataset":dname,"task":task,"seed":seed,"model":"TabPFN","time_s":round(dt,2),
                     "n_train":len(Xtr),"n_test":len(Xte),"n_cat":len(cat_cols),"n_num":len(num_cols),
                     "accuracy":accuracy_score(yte,pred),"bal_acc":balanced_accuracy_score(yte_e,le.transform(pred)),
                     "f1_macro":f1_score(yte_e,le.transform(pred),average="macro",zero_division=0),
                     "logloss":log_loss(yte_e,proba,labels=list(range(ncl))),"ece":ece_score(yte_e,proba)}
                try:
                    rec["auc"]=roc_auc_score(yte_e,proba[:,1]) if ncl==2 else roc_auc_score(yte_e,proba,multi_class="ovr")
                except: rec["auc"]=np.nan
            else:
                from tabpfn import TabPFNRegressor
                m=TabPFNRegressor(n_estimators=4, random_state=0)
                m.fit(Xt_tr, ytr.astype(float)); pred=m.predict(Xt_te); dt=time.time()-t0
                rec={"dataset":dname,"task":task,"seed":seed,"model":"TabPFN","time_s":round(dt,2),
                     "n_train":len(Xtr),"n_test":len(Xte),"n_cat":len(cat_cols),"n_num":len(num_cols),
                     "rmse":float(np.sqrt(mean_squared_error(yte.astype(float),pred))),
                     "mae":float(mean_absolute_error(yte.astype(float),pred)),
                     "r2":float(r2_score(yte.astype(float),pred))}
            rows.append(rec)
            print(f"  {dname} seed{seed}: {dt:.1f}s " + (f"acc={rec.get('accuracy',0):.3f}" if task=="class" else f"rmse={rec.get('rmse',0):.3f}"), flush=True)
        except Exception as e:
            print(f"  {dname} seed{seed} FAILED: {e}"); traceback.print_exc()
            rows.append({"dataset":dname,"task":task,"seed":seed,"model":"TabPFN","error":str(e)[:200]})
    out=pd.DataFrame(rows)
    # align to union schema so classification + regression appends share columns
    cols=["dataset","task","seed","model","time_s","n_train","n_test","n_cat","n_num",
          "accuracy","bal_acc","f1_macro","logloss","auc","ece","rmse","mae","r2","error"]
    out=out.reindex(columns=cols)
    fp=os.path.join(PY,"results_tabpfn.csv")
    if os.path.exists(fp):
        prev=pd.read_csv(fp).reindex(columns=cols)
        prev=prev[~((prev.dataset==dname)&(prev.model=="TabPFN"))]
        out=pd.concat([prev,out],ignore_index=True)
        out.to_csv(fp,index=False)
    else:
        out.to_csv(fp,index=False)
    print(f"appended {len(out)} rows to results_tabpfn.csv")

if __name__=="__main__":
    run_dataset(sys.argv[1] if len(sys.argv)>1 else "diabetes")
