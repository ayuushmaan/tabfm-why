"""Mechanistic probes: interactions (XOR vs linear), cat/num ablation, missingness,
calibration, permutation attribution agreement, shortcut exploitation, subgroup errors."""
import warnings, time, traceback
warnings.filterwarnings("ignore")
import webbrowser as _wb
_wb.open = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("browser blocked headless"))
import numpy as np, pandas as pd, os, sys
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

def get_clf(name):
    if name=="LightGBM": return lgb.LGBMClassifier(verbosity=-1, n_estimators=200, learning_rate=0.05, random_state=0)
    if name=="RandomForest": return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
    if name=="Linear": return LogisticRegression(max_iter=2000)
    if name=="MLP": return MLPClassifier(hidden_layer_sizes=(128,64), early_stopping=True, max_iter=500, random_state=0)
    if name=="TabPFN":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(n_estimators=4, random_state=0)
    if name=="TabICL":
        from tabicl import TabICLClassifier
        return TabICLClassifier(random_state=0)

MODELS = ["TabPFN","TabICL","LightGBM","RandomForest","Linear","MLP"]

def prep_fit_predict(mname, Xtr, ytr, Xte, cat_cols, num_cols):
    m = get_clf(mname)
    if mname in ("TabPFN","TabICL"):
        a,b = Xtr.copy(), Xte.copy()
        for c in cat_cols:
            a[c]=a[c].astype("category"); b[c]=b[c].astype("category")
        for c in num_cols:
            med=a[c].median(); a[c]=a[c].fillna(med); b[c]=b[c].fillna(med)
        m.fit(a,ytr); return m.predict(b), (m.predict_proba(b) if hasattr(m,"predict_proba") else None), m
    # pipeline
    if cat_cols and num_cols:
        prep = ColumnTransformer([
            ("num", Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]) if mname in ("Linear","MLP") else SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ord",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))] + ([("sc",StandardScaler())] if mname in ("Linear","MLP") else [])), cat_cols)])
    elif cat_cols:
        prep = Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ord",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))])
    else:
        prep = Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]) if mname in ("Linear","MLP") else SimpleImputer(strategy="median")
    pipe = Pipeline([("prep",prep),("m",m)])
    # sklearn MLP with early_stopping cannot handle string labels -> encode, then decode
    if mname == "MLP":
        le = LabelEncoder().fit(ytr)
        pipe.fit(Xtr, le.transform(pd.Series(ytr).reset_index(drop=True)))
        pred_e = pipe.predict(Xte)
        pred = le.inverse_transform(pred_e)
        proba = pipe.predict_proba(Xte) if hasattr(pipe,"predict_proba") else None
        pipe.le_ = le
        return pred, proba, pipe
    pipe.fit(Xtr,ytr)
    proba = pipe.predict_proba(Xte) if hasattr(pipe,"predict_proba") else None
    return pipe.predict(Xte), proba, pipe

def ece(y,p,n_bins=10):
    y=np.asarray(y); p=np.asarray(p)
    conf=p.max(1); pred=p.argmax(1); acc=(pred==y)
    edges=np.linspace(0,1,n_bins+1); s=0
    for i in range(n_bins):
        msk=(conf>edges[i])&(conf<=edges[i+1])
        if msk.sum()>0: s+=abs(acc[msk].mean()-conf[msk].mean())*msk.mean()
    return float(s)

# ---- 1. Interaction probe: XOR vs Linear synthetic ----
def probe_interaction():
    rng=np.random.RandomState(0)
    n=1200
    Xlin=rng.randn(n,6); ylin=(Xlin[:,0]+Xlin[:,1]-Xlin[:,2]>0).astype(int)
    Xxor=rng.randn(n,6); yxor=((Xxor[:,0]>0)^(Xxor[:,1]>0)).astype(int)
    recs=[]
    for dname,(X,y) in {"linear":(Xlin,ylin),"xor":(Xxor,yxor)}.items():
        X=pd.DataFrame(X,columns=[f"f{i}" for i in range(6)])
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
        for mname in MODELS:
            try:
                pred,_,_=prep_fit_predict(mname,Xtr,ytr,Xte,[],list(X.columns))
                recs.append({"probe":"interaction","setting":dname,"model":mname,"acc":accuracy_score(yte,pred)})
            except Exception as e: recs.append({"probe":"interaction","setting":dname,"model":mname,"acc":np.nan,"err":str(e)[:120]})
    return pd.DataFrame(recs)

# ---- 2. cat/num ablation on credit-g ----
def probe_catnum():
    b=fetch_openml(data_id=31,as_frame=True,parser="auto")
    X=b.data; y=pd.Series(b.target).astype(str)
    cat=[c for c in X.columns if str(X[c].dtype) in ("object","category") or X[c].dtype==object]
    num=[c for c in X.columns if c not in cat]
    for c in num: X[c]=pd.to_numeric(X[c],errors="coerce")
    recs=[]
    for setting,cols,cc in [("full",list(X.columns),cat),("num-only",num,[]),("cat-only",cat,cat if len(cat)>0 else [])]:
        Xs=X[cols]
        Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.25,random_state=0,stratify=y)
        nc=[c for c in cc if c in Xs.columns]; nn=[c for c in Xs.columns if c not in nc]
        for mname in MODELS:
            try:
                pred,_,_=prep_fit_predict(mname,Xtr,ytr,Xte,nc,nn)
                recs.append({"probe":"catnum","setting":setting,"model":mname,"acc":accuracy_score(yte,pred)})
            except Exception as e: recs.append({"probe":"catnum","setting":setting,"model":mname,"acc":np.nan,"err":str(e)[:120]})
    return pd.DataFrame(recs)

# ---- 3. missingness robustness on diabetes ----
def probe_missing():
    b=fetch_openml(data_id=37,as_frame=True,parser="auto")
    X=b.data; y=pd.Series(b.target).astype(str)
    for c in X.columns: X[c]=pd.to_numeric(X[c],errors="coerce")
    recs=[]
    for rate in [0.0,0.1,0.2,0.3]:
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
        rng=np.random.RandomState(int(rate*100))
        def corrupt(D):
            D=D.copy()
            mask=rng.rand(*D.shape)<rate
            D[mask]=np.nan
            return D
        Xtr_c, Xte_c = corrupt(Xtr), corrupt(Xte)
        for mname in MODELS:
            try:
                pred,_,_=prep_fit_predict(mname,Xtr_c,ytr,Xte_c,[],list(X.columns))
                recs.append({"probe":"missing","setting":f"mcar{int(rate*100)}","model":mname,"acc":accuracy_score(yte,pred)})
            except Exception as e: recs.append({"probe":"missing","setting":f"mcar{int(rate*100)}","model":mname,"acc":np.nan,"err":str(e)[:120]})
    return pd.DataFrame(recs)

# ---- 4. calibration on diabetes ----
def probe_calibration():
    b=fetch_openml(data_id=37,as_frame=True,parser="auto")
    X=b.data; y=pd.Series(b.target).astype(str)
    for c in X.columns: X[c]=pd.to_numeric(X[c],errors="coerce")
    le=LabelEncoder().fit(y); ye=le.transform(y)
    Xtr,Xte,ytr,yte=train_test_split(X,ye,test_size=0.3,random_state=0,stratify=ye)
    recs=[]
    for mname in MODELS:
        try:
            pred,proba,_=prep_fit_predict(mname,Xtr,ytr,Xte,[],list(X.columns))
            ll=log_loss(yte,proba) if proba is not None else np.nan
            recs.append({"probe":"calib","setting":"diabetes","model":mname,"acc":accuracy_score(yte,pred),"logloss":ll,"ece":ece(yte,proba) if proba is not None else np.nan})
        except Exception as e: recs.append({"probe":"calib","setting":"diabetes","model":mname,"acc":np.nan,"err":str(e)[:120]})
    return pd.DataFrame(recs)

# ---- 5. permutation attribution agreement on credit-g (subsampled) ----
def probe_attribution():
    b=fetch_openml(data_id=31,as_frame=True,parser="auto")
    X=b.data; y=pd.Series(b.target).astype(str)
    cat=[c for c in X.columns if str(X[c].dtype) in ("object","category") or X[c].dtype==object]
    num=[c for c in X.columns if c not in cat]
    for c in num: X[c]=pd.to_numeric(X[c],errors="coerce")
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
    # baseline acc then permute each col
    import collections
    imps={}
    for mname in MODELS:
        try:
            pred,_,fit = prep_fit_predict(mname,Xtr,ytr,Xte,cat,num)
            # need predict fn on mutated Xte
            base=accuracy_score(yte,pred)
            drops={}
            for c in X.columns[:10]:  # first 10 for speed
                Xp=Xte.copy()
                Xp[c]=np.random.RandomState(0).permutation(Xp[c].values)
                try:
                    if mname in ("TabPFN","TabICL"):
                        for cc in cat: Xp[cc]=Xp[cc].astype("category")
                        pp=fit.predict(Xp)
                    else:
                        pp=fit.predict(Xp)
                        _le = getattr(fit, "le_", None)
                        if _le is not None:
                            pp = _le.inverse_transform(pp)
                    drops[c]=base-accuracy_score(yte,pp)
                except: drops[c]=np.nan
            imps[mname]=drops
        except Exception as e: imps[mname]={}
    # pairwise spearman of importance vectors
    from scipy.stats import spearmanr
    keys=list(imps.keys()); rows=[]
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            a,b=imps[keys[i]],imps[keys[j]]
            common=[c for c in a if c in b and np.isfinite(a[c]) and np.isfinite(b[c])]
            r=np.nan
            if len(common)>=3:
                try: r=spearmanr([a[c] for c in common],[b[c] for c in common]).statistic
                except: pass
            rows.append({"probe":"attrib_agree","setting":f"{keys[i]}-vs-{keys[j]}","model":f"{keys[i]}|{keys[j]}","acc":r})
    imp_df=pd.DataFrame([{"model":k,"feat":f,"drop":v} for k,v in imps.items() for f,v in v.items()])
    return pd.DataFrame(rows), imp_df

# ---- 6. shortcut exploitation ----
def probe_shortcut():
    b=fetch_openml(data_id=37,as_frame=True,parser="auto")
    X=b.data; y=pd.Series(b.target).astype(str)
    for c in X.columns: X[c]=pd.to_numeric(X[c],errors="coerce")
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
    rng=np.random.RandomState(0)
    # spurious col: 90% correlated in train, random in test
    def spur(ys, corr):
        vals = pd.unique(ys)
        s=[]
        for v in ys:
            s.append(v if rng.rand()<corr else (np.random.choice(vals)))
        return np.array(s)
    Xtr_s=Xtr.copy(); Xte_s=Xte.copy()
    Xtr_s["SPUR"]=spur(ytr.values,0.9); Xte_s["SPUR"]=spur(yte.values,0.5)
    recs=[]
    for mname in MODELS:
        try:
            # clean baseline
            p0,_,_=prep_fit_predict(mname,Xtr,ytr,Xte,[],list(X.columns))
            p1,_,_=prep_fit_predict(mname,Xtr_s,ytr,Xte_s,["SPUR"],list(X.columns))
            a0,a1=accuracy_score(yte,p0),accuracy_score(yte,p1)
            recs.append({"probe":"shortcut","setting":"spur0.9-train","model":mname,"acc":a1,"drop_vs_clean":a1-a0,"clean":a0})
        except Exception as e: recs.append({"probe":"shortcut","setting":"spur","model":mname,"acc":np.nan,"err":str(e)[:120]})
    return pd.DataFrame(recs)

# ---- 7. subgroup errors: per-class recall on vehicle ----
def probe_subgroup():
    b=fetch_openml(data_id=54,as_frame=True,parser="auto")
    X=b.data; y=pd.Series(b.target).astype(str)
    for c in X.columns: X[c]=pd.to_numeric(X[c],errors="coerce")
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
    recs=[]
    for mname in MODELS:
        try:
            pred,_,_=prep_fit_predict(mname,Xtr,ytr,Xte,[],list(X.columns))
            for cls in sorted(y.unique()):
                msk=(yte==cls)
                recs.append({"probe":"subgroup","setting":f"vehicle:{cls}","model":mname,"acc":accuracy_score(yte[msk],pred[msk])})
        except Exception as e: recs.append({"probe":"subgroup","setting":"vehicle","model":mname,"acc":np.nan})
    return pd.DataFrame(recs)

if __name__=="__main__":
    which = sys.argv[1] if len(sys.argv)>1 else "all"
    allp=[]
    def save(tag, df):
        fp=os.path.join(RESULTS,f"results_mech_{tag}.csv")
        df.to_csv(fp,index=False); print(f"Saved {fp}", df.shape)
    if which in ("all","part1"):
        print("probe: interaction",flush=True); d=probe_interaction(); print(d); allp.append(d); save("interaction",d)
        print("probe: catnum",flush=True); d=probe_catnum(); print(d); allp.append(d); save("catnum",d)
        print("probe: missing",flush=True); d=probe_missing(); print(d); allp.append(d); save("missing",d)
        print("probe: calibration",flush=True); d=probe_calibration(); print(d); allp.append(d); save("calib",d)
    if which in ("all","part2"):
        print("probe: attribution",flush=True)
        try:
            agree,impd=probe_attribution(); print(agree); print(impd.head(20)); allp.append(agree)
            save("attrib_agree",agree); impd.to_csv(os.path.join(RESULTS,"results_attrib_importance.csv"),index=False)
        except Exception as e: print("attrib failed",e); traceback.print_exc()
    if which in ("all","part2","part3"):
        print("probe: shortcut",flush=True); d=probe_shortcut(); print(d); allp.append(d); save("shortcut",d)
        print("probe: subgroup",flush=True); d=probe_subgroup(); print(d); allp.append(d); save("subgroup",d)
    if allp:
        out=pd.concat(allp,ignore_index=True)
        out.to_csv(os.path.join(RESULTS,f"results_mechanistic_{which}.csv"),index=False)
        print(f"Saved results_mechanistic_{which}.csv", out.shape)
