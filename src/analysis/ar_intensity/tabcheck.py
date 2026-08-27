"""Definitive tab:concepts check: distinct nodes, firing events, med IVT, %>=250/500,
and spatio-temporal containment P(par|ch)+lift for all 8 concepts, one pass at THRESH=0."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
N=int(os.environ.get("NMAX","8000")); THRESH=float(os.environ.get("THRESH","0.0"))
DEV="cuda" if torch.cuda.is_available() else "cpu"; OUT="/scratch/euh7ys/climate_xai/plots"; CKPT=f"{OUT}/tabcheck.ckpt.npz"
FAM={340:[3481,3948,3675], 99:[1454,3392,2722]}; CC=[c for p in FAM for c in [p]+FAM[p]]; CH=[x for p in FAM for x in FAM[p]]
EDG=np.arange(0,2001,5.0)
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy",""),"%Y-%m-%dT%H-%M")
def main():
    print("dev",DEV,"N",N,"THRESH",THRESH,flush=True)
    idx,levels,qi,ui,vi,lat_i,lon_i=load_channel_index()
    m,c,fmin,frng=load_sae("matry_L8",DEV); CCt=torch.as_tensor(CC,device=DEV)
    files=sorted(glob.glob(f"{c['act']}/layer0008_*.npy")); sel=files if N>=len(files) else [files[i] for i in np.linspace(0,len(files)-1,N).astype(int)]
    ncnt=None; ne={cc:0 for cc in CC}; ih={cc:np.zeros(len(EDG)+1,np.int64) for cc in CC}
    g250={cc:0 for cc in CC}; g500={cc:0 for cc in CC}; cof={x:0 for x in CH}; pne={p:0 for p in FAM}; ngrid=0; nts=0; start=0
    if os.path.exists(CKPT):
        z=np.load(CKPT); start=int(z["j"][0]); ngrid=int(z["ngrid"][0]); nts=int(z["nts"][0])
        ncnt={cc:z[f"ncnt_{cc}"].copy() for cc in CC}
        for cc in CC: ne[cc]=int(z[f"ne_{cc}"][0]); ih[cc]=z[f"ih_{cc}"].copy(); g250[cc]=int(z[f"g250_{cc}"][0]); g500[cc]=int(z[f"g500_{cc}"][0])
        for x in CH: cof[x]=int(z[f"cof_{x}"][0])
        for p in FAM: pne[p]=int(z[f"pne_{p}"][0])
        print("RESUMED",start,flush=True)
    def ck(j):
        d={"j":np.array([j]),"ngrid":np.array([ngrid]),"nts":np.array([nts])}
        for cc in CC: d[f"ncnt_{cc}"]=ncnt[cc]; d[f"ne_{cc}"]=np.array([ne[cc]]); d[f"ih_{cc}"]=ih[cc]; d[f"g250_{cc}"]=np.array([g250[cc]]); d[f"g500_{cc}"]=np.array([g500[cc]])
        for x in CH: d[f"cof_{x}"]=np.array([cof[x]])
        for p in FAM: d[f"pne_{p}"]=np.array([pne[p]])
        np.savez(CKPT,**d)
    for k in range(start,len(sel)):
        dt=pdt(os.path.basename(sel[k]))
        try: era=np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        except FileNotFoundError: continue
        iv=node_ivt(era,qi,ui,vi,levels)
        a=np.load(sel[k],mmap_mode="r"); x=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
        if fmin is not None: x=(2.0*(x-fmin)/frng-1.0).astype(np.float32)
        with torch.no_grad(): fb=(encode(m,c["arch"],torch.from_numpy(x).to(DEV))[:,CCt]>THRESH).cpu().numpy()
        if ncnt is None: ncnt={cc:np.zeros(fb.shape[0],np.int64) for cc in CC}
        fr={cc:fb[:,i] for i,cc in enumerate(CC)}
        for cc in CC:
            fm=fr[cc]; ncnt[cc]+=fm; ne[cc]+=int(fm.sum()); ivc=iv[fm]
            ih[cc]+=np.bincount(np.digitize(ivc,EDG),minlength=len(EDG)+1); g250[cc]+=int((ivc>=250).sum()); g500[cc]+=int((ivc>=500).sum())
        for p,chs in FAM.items():
            pne[p]+=int(fr[p].sum())
            for x2 in chs: cof[x2]+=int((fr[x2]&fr[p]).sum())
        ngrid+=fb.shape[0]; nts+=1
        if (k+1)%1000==0: ck(k+1); print(f"  {k+1}/{len(sel)} ck",flush=True)
    def md(cc):
        h=ih[cc]; t=h.sum();
        if t==0: return 0.0
        i=np.searchsorted(np.cumsum(h),t/2); return float(EDG[min(max(i-1,0),len(EDG)-1)])
    print(f"\n== tab:concepts check  THRESH={THRESH}  timesteps={nts} ==")
    print(f"{'con':>6}{'nodes':>8}{'events':>11}{'medIVT':>8}{'>=250':>7}{'>=500':>7}{'P(par|ch)':>11}{'lift':>7}")
    for p,chs in FAM.items():
        Pp=pne[p]/ngrid
        print(f"{p:>6}{int((ncnt[p]>0).sum()):>8}{ne[p]:>11}{md(p):>8.0f}{100*g250[p]/max(ne[p],1):>6.0f}%{100*g500[p]/max(ne[p],1):>6.0f}%{'-':>11}{'-':>7}  parent P(par)={Pp:.4g}")
        for x2 in chs:
            T=cof[x2]/max(ne[x2],1)
            print(f"{x2:>6}{int((ncnt[x2]>0).sum()):>8}{ne[x2]:>11}{md(x2):>8.0f}{100*g250[x2]/max(ne[x2],1):>6.0f}%{100*g500[x2]/max(ne[x2],1):>6.0f}%{T:>11.3f}{(T/Pp if Pp else 0):>7.0f}")
    if os.path.exists(CKPT): os.remove(CKPT)
    print("DONE",flush=True)
if __name__=="__main__": main()
