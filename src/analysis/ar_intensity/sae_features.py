"""Encode SAE concept vectors per qualifying AR: per-region mesh-node activation rate."""
import numpy as np, torch
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime
from src.analysis.ar_intensity.ivt_pipeline import region_node_setup
from src.models.plain_sae import PlainSAE, topk
from src.models.matryoshka_sae import MatryoshkaSAE
P="/project/AikyamLab/madelyn/GraphCast"; SCR="/scratch/euh7ys/climate_xai"
SAES={
 "plain_L0": dict(arch="plain",layer=0, ckpt=f"{P}/train/PlainSAE/Layer00/final_model.pt",      act=f"{SCR}/activations/layer00"),
 "plain_L8": dict(arch="plain",layer=8, ckpt=f"{SCR}/checkpoints/plain_layer8/final_model.pt",   act=f"{P}/activations/Layer08"),
 "plain_L15":dict(arch="plain",layer=15,ckpt=f"{P}/train/PlainSAE/Layer15/checkpoint_epoch008.pt",act=f"{P}/activations/Layer15"),
 "matry_L0": dict(arch="matry",layer=0, ckpt=f"{P}/train/MatryoshkaSAE/Layer00/final_model.pt",  act=f"{SCR}/activations/layer00"),
 "matry_L8": dict(arch="matry",layer=8, ckpt=f"{P}/train/MatryoshkaSAE/Layer08/final_model.pt",  act=f"{P}/activations/Layer08"),
 "matry_L15":dict(arch="matry",layer=15,ckpt=f"{P}/train/MatryoshkaSAE/Layer15/final_model.pt",  act=f"{P}/activations/Layer15"),
}
def load_sae(name, dev):
    c=SAES[name]; st=torch.load(c["ckpt"],map_location=dev); st=st.get("model_state_dict",st) if isinstance(st,dict) else st
    if c["arch"]=="plain":
        m=PlainSAE(d_in=512,n_latents=4096); fmin=frng=None
    else:
        m=MatryoshkaSAE(512,4096,[256,512,1024,2048,4096],32,300000,topk_mode="per_sample")
        fmin=np.load(f"{c['act']}/feature_min.npy").astype(np.float32)
        fmax=np.load(f"{c['act']}/feature_max.npy").astype(np.float32); frng=fmax-fmin; frng[frng<1e-8]=1.0
    m.load_state_dict(st,strict=False); m.to(dev).eval()
    return m, c, fmin, frng
def encode(m, arch, x):
    if arch=="plain":
        xn=x-x.mean(1,keepdim=True); xn=xn/xn.norm(dim=1,keepdim=True).clamp_min(1e-6)
        return topk(torch.relu(m.enc(xn-m.b_pre)), m.k_active)
    return m.get_acts(x, indices=None, normalize=True)
def act_file(c, dt):
    return f"{c['act']}/layer{c['layer']:04d}_mesh_gnn_post_res_nodes_mesh_nodes_t{dt.strftime('%Y-%m-%dT%H-%M')}.npy"
def build_features(name, time_region_df, dev="cpu", progress=0):
    m,c,fmin,frng=load_sae(name,dev); setup=region_node_setup()
    unodes=np.unique(np.concatenate([setup[r]["nodes"] for r in REGIONS]))
    pos={int(n):i for i,n in enumerate(unodes)}
    rmap={r:np.array([pos[int(n)] for n in setup[r]["nodes"]]) for r in REGIONS}
    feats=[]; meta=[]; n=0
    for ti,grp in time_region_df.groupby("time_index"):
        dt=index_to_datetime(int(ti))
        try: a=np.load(act_file(c,dt), mmap_mode="r")
        except Exception: continue
        x=np.ascontiguousarray(a[unodes]).astype(np.float32)
        if fmin is not None: x=(2.0*(x-fmin)/frng-1.0).astype(np.float32)
        with torch.no_grad():
            fired=(encode(m,c["arch"],torch.from_numpy(x).to(dev))>0).float().cpu().numpy()
        for r in grp.region:
            feats.append(fired[rmap[r]].mean(0)); meta.append((int(ti),r))
        n+=1
        if progress and n%progress==0: print(f"  {name}: {n} timesteps",flush=True)
    return np.array(feats,dtype=np.float32), meta
