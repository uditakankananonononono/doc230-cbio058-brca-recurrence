import os,sys,json,time,tarfile,numpy as np,pandas as pd,torch
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,accuracy_score
torch.set_num_threads(2); D='data/brca_metabric/'; OUT='results/'; os.makedirs(OUT,exist_ok=True)
def log(*a): print(time.strftime('%H:%M:%S'),*a,flush=True)
# ---------------- prep (no outcome modelling)
def prep():
    if os.path.exists('data/prep.npz'): return np.load('data/prep.npz',allow_pickle=True)
    pt=pd.read_csv(D+'data_clinical_patient.txt',sep='\t',comment='#').set_index('PATIENT_ID')
    sm=pd.read_csv(D+'data_clinical_sample.txt',sep='\t',comment='#').set_index('PATIENT_ID')
    def readmat(fn):
        with open(D+fn) as f:
            hdr=f.readline().rstrip('\n').split('\t'); cols=hdr[2:]; names=[];rows=[]
            for line in f:
                p=line.rstrip('\n').split('\t'); names.append(p[0]); rows.append(np.array([float(v) if v not in ('','NA') else np.nan for v in p[2:]],dtype='float32'))
        M=np.vstack(rows); df=pd.DataFrame(M,index=names,columns=cols); df=df[~df.index.duplicated()]; return df
    ex=readmat('data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt'); cn=readmat('data_cna.txt')
    ids=[i for i in pt.index if i in ex.columns and i in cn.columns and i in sm.index and str(pt.loc[i,'RFS_STATUS'])[:1] in '01' and str(pt.loc[i,'RFS_STATUS'])!='nan' and sm.loc[i,'ER_STATUS'] in ('Positive','Negative')]
    pt=pt.loc[ids];sm=sm.loc[ids]
    y=np.array([int(s[0]) for s in pt.RFS_STATUS]); c=(sm.ER_STATUS=='Positive').values.astype(np.float32)
    Xe=np.nan_to_num(ex[ids].values.T,nan=0.0).astype('float32'); Xc=np.nan_to_num(cn[ids].values.T,nan=0.0).astype('float32')
    num=lambda s:pd.to_numeric(s,errors='coerce').values.astype('float64')
    clin=np.column_stack([num(pt.AGE_AT_DIAGNOSIS),num(pt.LYMPH_NODES_EXAMINED_POSITIVE),num(sm.TUMOR_SIZE),num(sm.TUMOR_STAGE),num(sm.GRADE),num(pt.NPI),c,
        (sm.HER2_STATUS=='Positive').values*1.0,(sm.PR_STATUS=='Positive').values*1.0])
    treat=np.column_stack([(pt.CHEMOTHERAPY=='YES')*1.0,(pt.HORMONE_THERAPY=='YES')*1.0,(pt.RADIO_THERAPY=='YES')*1.0,(pt.BREAST_SURGERY=='MASTECTOMY')*1.0,(pt.BREAST_SURGERY=='BREAST CONSERVING')*1.0])
    np.savez('data/prep.npz',ids=np.array(ids),y=y,c=c,Xe=Xe,Xc=Xc,genes_e=np.array(ex.index),clin=clin,treat=treat,
        rfs=num(pt.RFS_MONTHS),cohort=num(pt.COHORT),hormone=(pt.HORMONE_THERAPY=='YES').values*1)
    return np.load('data/prep.npz',allow_pickle=True)
# ---------------- model
def topvar(X,tr,k=2000): return np.argsort(-X[tr].var(0))[:k]
class AE(torch.nn.Module):
    def __init__(s,de,dc,adj,mode):
        super().__init__(); s.mode=mode; s.adj=adj; h=256;z=32
        s.ee=torch.nn.Sequential(torch.nn.Linear(de,h),torch.nn.ReLU(),torch.nn.Linear(h,z)); s.ec=torch.nn.Sequential(torch.nn.Linear(dc,h),torch.nn.ReLU(),torch.nn.Linear(h,z))
        zin=z+(1 if adj else 0)
        s.de=torch.nn.Sequential(torch.nn.Linear(zin,h),torch.nn.ReLU(),torch.nn.Linear(h,de)); s.dc=torch.nn.Sequential(torch.nn.Linear(zin,h),torch.nn.ReLU(),torch.nn.Linear(h,dc))
    def emb(s,xe,xc):
        if s.mode=='expr': return s.ee(xe),s.ee(xe),s.ee(xe)
        if s.mode=='cna': return s.ec(xc),s.ec(xc),s.ec(xc)
        a=s.ee(xe);b=s.ec(xc);return (a+b)/2,a,b
    def forward(s,xe,xc,c):
        z,a,b=s.emb(xe,xc); zz=torch.cat([z,c[:,None]],1) if s.adj else z
        return z,s.de(zz),s.dc(zz),a,b
def fit_embed(Xe,Xc,c,tr,others,adj=True,mode='both',epochs=None,seed=0):
    epochs=epochs or EPOCHS
    ge=topvar(Xe,tr);gc=topvar(Xc,tr)
    mu_e=Xe[tr][:,ge].mean(0);sd_e=Xe[tr][:,ge].std(0)+1e-6;mu_c=Xc[tr][:,gc].mean(0);sd_c=Xc[tr][:,gc].std(0)+1e-6
    prep_=lambda idx:(torch.tensor((Xe[idx][:,ge]-mu_e)/sd_e),torch.tensor((Xc[idx][:,gc]-mu_c)/sd_c),torch.tensor(c[idx]))
    torch.manual_seed(seed); m=AE(len(ge),len(gc),adj,mode); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4)
    xe,xc,cc=prep_(tr); n=len(tr); g=torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm=torch.randperm(n,generator=g)
        for i in range(0,n,128):
            b=perm[i:i+128]; z,re,rc,a,bb=m(xe[b],xc[b],cc[b])
            loss=((re-xe[b])**2).mean()+((rc-xc[b])**2).mean()+(0.1*((a-bb)**2).mean() if mode=='both' else 0)
            opt.zero_grad();loss.backward();opt.step()
    m.eval(); outs=[]
    with torch.no_grad():
        for idx in [tr]+others:
            e,c_,cn_=prep_(idx); outs.append(m.emb(e,c_)[0].numpy())
    return outs,ge
def rf(): return RF(300,max_features='sqrt',n_jobs=2,random_state=0)
def folds(y,seed=0): return list(StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(y)),y))
def boot(y,p,B=2000):
    r=np.random.RandomState(0);n=len(y);v=[]
    for _ in range(B):
        i=r.randint(0,n,n)
        if len(set(y[i]))>1: v.append(roc_auc_score(y[i],p[i]))
    return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
def summ(y,p,fa,fc):
    return dict(auc_pooled=float(roc_auc_score(y,p)),auc_fold_mean=float(np.mean(fa)),auc_fold_sd=float(np.std(fa,ddof=1)),acc=float(np.mean(fc)),n=int(len(y)),pos=int(y.sum()))
CACHE={}
EPOCHS=60
def cv_aime(P,idx,tag,adj=True,mode='both',yv=None,leak=False,seed_f=0):
    """5-fold CV inside subset idx; embedding fitted on training fold (or on all idx if leak). yv overrides labels (permutation)."""
    y=(P['y'] if yv is None else yv)[idx]; F=folds(y,seed_f); pp=np.zeros(len(idx));fa=[];fc=[]
    for k,(tr,te) in enumerate(F):
        key=(tag,adj,mode,leak,k)
        if key not in CACHE:
            if leak: (zall,),_=fit_embed(P['Xe'],P['Xc'],P['c'],idx,[],adj,mode); CACHE[key]=(zall[tr],zall[te],None)
            else: (ztr,zte),ge=fit_embed(P['Xe'],P['Xc'],P['c'],idx[tr],[idx[te]],adj,mode); CACHE[key]=(ztr,zte,ge)
        ztr,zte,_=CACHE[key]; m=rf().fit(ztr,y[tr]); p=m.predict_proba(zte)[:,1]; pp[te]=p
        fa.append(roc_auc_score(y[te],p)); fc.append(accuracy_score(y[te],p>0.5))
    return summ(y,pp,fa,fc),pp
def cv_feat(X,idx,y,seed_f=0,sel=None):
    yy=y[idx];F=folds(yy,seed_f);pp=np.zeros(len(idx));fa=[];fc=[]
    for tr,te in F:
        Xtr,Xte=X[idx][tr],X[idx][te]
        if sel=='var': 
            ge=topvar(Xtr,np.arange(len(Xtr)),2000)
        med=np.nanmedian(Xtr,0);med=np.where(np.isnan(med),0,med); Xtr=np.where(np.isnan(Xtr),med,Xtr);Xte=np.where(np.isnan(Xte),med,Xte)
        m=rf().fit(Xtr,yy[tr]);p=m.predict_proba(Xte)[:,1];pp[te]=p;fa.append(roc_auc_score(yy[te],p));fc.append(accuracy_score(yy[te],p>0.5))
    return summ(yy,pp,fa,fc),pp
def raw_top(P,idx_all):  # raw baseline feature matrix: top var features chosen inside each fold -> implemented via cv_raw
    pass
def cv_raw(P,idx):
    y=P['y'][idx];F=folds(y);pp=np.zeros(len(idx));fa=[];fc=[]
    for tr,te in F:
        g=idx[tr];ge=topvar(P['Xe'],g);gc=topvar(P['Xc'],g)
        X=np.hstack([P['Xe'][idx][:,ge],P['Xc'][idx][:,gc]]); m=rf().fit(X[tr],y[tr]);p=m.predict_proba(X[te])[:,1];pp[te]=p
        fa.append(roc_auc_score(y[te],p));fc.append(accuracy_score(y[te],p>0.5))
    return summ(y,pp,fa,fc),pp
def main():
    global EPOCHS
    P=dict(prep())
    if os.environ.get('SMOKE'):  # crash test only: random labels, tiny subset, 1 epoch; outputs go to /tmp
        global OUT; OUT='/tmp/smoke058/'; os.makedirs(OUT,exist_ok=True); EPOCHS=1
        r=np.random.RandomState(1); sub=np.sort(r.choice(len(P['y']),400,replace=False))
        P={k:(v[sub] if getattr(v,'ndim',0)>=1 and len(v)==len(P['y']) and k not in ('genes_e',) else v) for k,v in P.items()}; P['y']=r.permutation(P['y'])
    n=len(P['y']); allidx=np.arange(n); res={}; y=P['y']; er=P['c']; rfs=P['rfs']
    # D1
    res['D1']=dict(n=int(n),recurred=int(y.sum()),er_pos=int(er.sum()),er_pos_recur_rate=float(y[er==1].mean()),er_neg_recur_rate=float(y[er==0].mean()),
        cohorts={str(int(k)):int((P['cohort']==k).sum()) for k in np.unique(P['cohort'][~np.isnan(P['cohort'])])},
        nonrecurred_followup_median=float(np.nanmedian(rfs[y==0])),nonrecurred_lt60=int(((y==0)&(rfs<60)).sum()),n_genes_expr=int(P['Xe'].shape[1]),n_genes_cna=int(P['Xc'].shape[1]))
    log('D1',res['D1'])
    # D2
    r2,p2=cv_aime(P,allidx,'all'); r2['auc_ci95']=boot(y,p2); r2['gate_predicts']=bool(r2['auc_pooled']>=0.70); res['D2']=r2; log('D2',r2)
    # D3
    b1,_=cv_raw(P,allidx); b3,_=cv_feat(P['clin'],allidx,y)
    res['D3']=dict(raw_rf=b1,clinical_rf=b3,diff_vs_best=float(r2['auc_pooled']-max(b1['auc_pooled'],b3['auc_pooled'])));res['D3']['gate_adds_value']=bool(res['D3']['diff_vs_best']>=0.02);log('D3',res['D3'])
    # D4
    a,_=cv_feat(er[:,None],allidx,y); b,_=cv_aime(P,allidx,'all',adj=False)
    res['D4']=dict(er_only=a,no_adjustment=b,strata={})
    for nm,v in (('ER+',1),('ER-',0)):
        ii=np.where(er==v)[0]; s,_=cv_aime(P,ii,'er'+str(v)); res['D4']['strata'][nm]=s
    res['D4']['gate_er_material']=bool(a['auc_pooled']>=0.55);res['D4']['gate_adjustment_matters']=bool(abs(r2['auc_pooled']-b['auc_pooled'])>=0.02);log('D4',res['D4'])
    # D5
    keep=np.where((y==1)|(rfs>=60))[0]; d5,_=cv_aime(P,keep,'lt60'); c5,_=cv_feat(P['clin'],keep,y)
    fu,_=cv_feat(np.nan_to_num(rfs)[:,None],allidx,y)
    res['D5']=dict(n_kept=int(len(keep)),aime=d5,clinical=c5,followup_only_all=fu,delta_aime=float(d5['auc_pooled']-r2['auc_pooled']),delta_clin=float(c5['auc_pooled']-b3['auc_pooled']));res['D5']['gate_censoring_sensitive']=bool(abs(res['D5']['delta_aime'])>=0.05);log('D5',res['D5'])
    # D6
    t,_=cv_feat(P['treat'],allidx,y); res['D6']=dict(treatment_only=t,strata={})
    for nm,v in (('hormone_YES',1),('hormone_NO',0)):
        ii=np.where(P['hormone']==v)[0]; s,_=cv_aime(P,ii,'h'+str(v)); res['D6']['strata'][nm]=s
    res['D6']['gate_treatment_material']=bool(t['auc_pooled']>=0.55);log('D6',res['D6'])
    # D7
    l,_=cv_aime(P,allidx,'all',leak=True); res['D7']=dict(leaky=l,diff=float(l['auc_pooled']-r2['auc_pooled']));res['D7']['gate_leakage_matters']=bool(res['D7']['diff']>=0.02);log('D7',res['D7'])
    # D8
    coh=P['cohort']; ok=~np.isnan(coh); iic=np.where(ok)[0]; yc=coh[iic].astype(int);Fc=folds(yc if np.bincount(yc).min()>=5 else (yc>0).astype(int))
    ca=[];pr=np.zeros(len(iic),int)
    # cohort identity from embeddings of the D2 folds (fold embeddings on allidx; map rows)
    F=folds(y)
    for k,(tr,te) in enumerate(F):
        ztr,zte,_=CACHE[('all',True,'both',False,k)]; m=rf().fit(ztr,coh[tr].astype(int) if not np.isnan(coh[tr]).any() else np.nan_to_num(coh[tr]).astype(int))
        pp=m.predict(zte); ca.append(float((pp==np.nan_to_num(coh[te]).astype(int))[ok[te]].mean()))
    maj=float(np.bincount(coh[ok].astype(int)).max()/ok.sum())
    big=[int(k) for k in np.unique(coh[ok]) if (coh==k).sum()>=100]; loco={}
    for k in big:
        te=np.where(coh==k)[0];tr=np.where((coh!=k)&ok)[0]
        (ztr,zte),_=fit_embed(P['Xe'],P['Xc'],P['c'],tr,[te]); m=rf().fit(ztr,y[tr]); loco[str(k)]=float(roc_auc_score(y[te],m.predict_proba(zte)[:,1]))
    res['D8']=dict(cohort_identity_acc=float(np.mean(ca)),majority=maj,loco_auc=loco,loco_mean=float(np.mean(list(loco.values()))));res['D8']['gate_cohort_dependent']=bool(res['D8']['loco_mean']<r2['auc_pooled']-0.05);log('D8',res['D8'])
    # D9
    perm=[]
    for s in range(5):
        yp=np.random.RandomState(s).permutation(y); r,_=cv_aime(P,allidx,'all',yv=yp); perm.append(r['auc_pooled'])
    res['D9']=dict(perm_auc=perm,perm_mean=float(np.mean(perm)));res['D9']['null_sane']=bool(abs(res['D9']['perm_mean']-0.5)<=0.03)
    F=folds(y);lists=[];top_auc=[];rand_auc=[]
    for k,(tr,te) in enumerate(F):
        ztr,zte,ge=CACHE[('all',True,'both',False,k)]; m=rf().fit(ztr,y[tr]); imp=m.feature_importances_
        Xs=P['Xe'][:,topvar(P['Xe'],tr)]  # expression genes selected in training fold (same rule as the AE input)
        Xtr=Xs[tr];corr=np.abs(np.corrcoef(Xtr.T,ztr.T)[:Xtr.shape[1],Xtr.shape[1]:]);corr=np.nan_to_num(corr)
        score=corr@imp; top=np.argsort(-score)[:25]; lists.append(set(top.tolist()))
        mm=rf().fit(Xtr[:,top],y[tr]);top_auc.append(roc_auc_score(y[te],mm.predict_proba(Xs[te][:,top])[:,1]))
        r=np.random.RandomState(k); ra=[]
        for d in range(10):
            g=r.choice(Xs.shape[1],25,replace=False); mm=rf().fit(Xtr[:,g],y[tr]); ra.append(roc_auc_score(y[te],mm.predict_proba(Xs[te][:,g])[:,1]))
        rand_auc.append(np.mean(ra))
    jac=[len(a&b)/len(a|b) for i,a in enumerate(lists) for b in lists[i+1:]]
    res['D9'].update(top25_auc_fold_mean=float(np.mean(top_auc)),random25_auc_fold_mean=float(np.mean(rand_auc)),jaccard_mean=float(np.mean(jac)))
    res['D9']['gate_list_specific']=bool(np.mean(top_auc)-np.mean(rand_auc)>=0.05);res['D9']['gate_list_stable']=bool(np.mean(jac)>=0.30);log('D9',res['D9'])
    # D10
    res['D10']={m:cv_aime(P,allidx,'all',mode=m)[0] for m in ('expr','cna')};log('D10',res['D10'])
    json.dump(res,open(OUT+'results058.json','w'),indent=1,default=float); log('DONE')
if __name__=='__main__': main()
