import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T="/scratch/euh7ys/climate_xai/concept_ivt"; PLOTS="/scratch/euh7ys/climate_xai/plots"
pairs=[(1592,"W_N_America"),(2948,"W_Europe"),(3218,"W_S_America"),(3720,"E_Australia")]
seas=[("DJF",[12,1,2]),("MAM",[3,4,5]),("JJA",[6,7,8]),("SON",[9,10,11]),("ALL",list(range(1,13)))]
def r(a,b): return float(np.corrcoef(a,b)[0,1]) if len(a)>2 else float("nan")
print("=== seasonal r(A_mean, IVT)  Standard SAE L8 ===")
print(f"{'concept / region':24}"+"".join(f"{s[0]:>7}" for s in seas))
store={}
for c,reg in pairs:
    d=np.load(f"{T}/track_pool_{reg}.npz"); mon=d["month"]; iv=d["ivt"].astype(float); am=d["A_mean"][:,c].astype(float); ok=np.isfinite(iv); del d
    store[c]=(mon,iv,am,ok)
    print(f"{str(c)+'  '+reg:24}"+"".join(f"{r(am[np.isin(mon,mm)&ok],iv[np.isin(mon,mm)&ok]):7.2f}" for _,mm in seas))
print("\n=== 3720 EAU: zero-inflation and conditional r (firing timesteps only) ===")
mon,iv,am,ok=store[3720]
print(f"{'season':6}{'zero%':>7}{'n_fire':>8}{'r_all':>7}{'r_fire':>8}")
for name,mm in seas:
    s=np.isin(mon,mm)&ok; f=s&(am>0)
    print(f"{name:6}{100*(am[s]==0).mean():6.1f}%{int(f.sum()):8d}{r(am[s],iv[s]):7.2f}{r(am[f],iv[f]):8.2f}")
# season-coloured scatter + binned-median overlay
fig,ax=plt.subplots(figsize=(7,5)); col={"DJF":"#d62728","MAM":"#ff7f0e","JJA":"#1f77b4","SON":"#2ca02c"}
for name,mm in seas[:4]:
    s=np.isin(mon,mm)&ok; ax.scatter(iv[s],am[s],s=2,alpha=0.06,c=col[name],rasterized=True)
top=np.percentile(iv[ok],99); bins=np.linspace(0,top,16); bc=(bins[:-1]+bins[1:])/2
for name,mm in seas[:4]:
    s=np.isin(mon,mm)&ok; x=iv[s]; y=am[s]; ix=np.digitize(x,bins)
    med=[np.median(y[ix==i]) if (ix==i).sum()>20 else np.nan for i in range(1,len(bins))]
    ax.plot(bc,med,c=col[name],lw=2.6,label=name)
ax.set_xlabel("IVT (kg m$^{-1}$ s$^{-1}$)"); ax.set_ylabel("latent 3720 activation")
ax.set_title("EAU latent 3720: activation vs IVT by season"); ax.legend()
fig.tight_layout(); fig.savefig(f"{PLOTS}/3720_eau_scatter.png",dpi=140); print("\nsaved",f"{PLOTS}/3720_eau_scatter.png")
