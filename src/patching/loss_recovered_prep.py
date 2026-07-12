"""Per event x (sae, layer): delta_recon = recon(x)-x in RAW activation space + delta_zero=-x.
Plain: forward gives normalized-space recon; raw recon = recon_n*||x-mean|| + mean.
Matry: minmax-scale -> normalize -> decode -> unnormalize -> un-scale."""
import os, glob, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
OUT = "/scratch/euh7ys/climate_xai/patching/lossrec"; os.makedirs(OUT, exist_ok=True)
EVENTS = ["1981-01-10T00-00","1981-04-18T12-00","1981-07-22T06-00","1981-10-05T18-00",
          "1987-01-15T12-00","1987-04-02T00-00","1987-07-20T12-00","1987-10-28T06-00",
          "1993-01-25T06-00","1993-04-11T18-00","1993-07-04T00-00","1993-10-16T12-00",
          "1999-02-08T00-00","1999-05-14T12-00","1999-08-19T06-00","1999-11-23T18-00",
          "2005-01-15T00-00","2005-06-09T12-00","2005-09-01T06-00","2005-12-25T18-00",
          "2011-03-05T00-00","2011-06-01T12-00","2014-11-20T18-00","2017-08-21T18-00"]
SAES = ["matry_L0","matry_L8","matry_L15","plain_L0","plain_L8","plain_L15"]
def main():
    for name in SAES:
        m, c, fmin, frng = load_sae(name, "cpu")
        files = sorted(glob.glob(f"{c['act']}/layer*_*.npy"))
        tags = {os.path.basename(f).split("_t")[-1].replace(".npy",""): f for f in files}
        for ev in EVENTS:
            f = tags.get(ev) or tags[min(tags, key=lambda t: abs(int(t[:4])*1e6+int(t[5:7])*1e4+int(t[8:10])*1e2+int(t[11:13]) - (int(ev[:4])*1e6+int(ev[5:7])*1e4+int(ev[8:10])*1e2+int(ev[11:13]))))]
            tag = os.path.basename(f).split("_t")[-1].replace(".npy","")
            a = np.load(f); xr_raw = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
            xt = torch.from_numpy(xr_raw)
            with torch.no_grad():
                if fmin is not None:   # matryoshka
                    fm = torch.from_numpy(np.asarray(fmin, np.float32)); fr = torch.from_numpy(np.asarray(frng, np.float32))
                    xs = 2.0*(xt - fm)/fr - 1.0
                    xn = m.normalizer.normalize(xs)
                    code = m._apply_topk(xn @ m.W_enc + m.b_enc, m.target_l0)
                    rec_s = m.normalizer.unnormalize(code @ m.W_dec + m.b_dec)   # 512-dim: exact inverse
                    rec = (rec_s + 1.0)*fr/2.0 + fm
                else:                  # plain: un-normalize forward's recon back to raw
                    out = m(xt); rec_n = out[0] if isinstance(out,(tuple,list)) else out
                    xm = xt - xt.mean(1, keepdim=True); rn = xm.norm(dim=1, keepdim=True).clamp_min(1e-6)
                    rec = rec_n * rn + xt.mean(1, keepdim=True)
                d = (rec - xt).numpy().astype(np.float32)
            np.save(f"{OUT}/drecon_{name}_{tag}.npy", d)
            np.save(f"{OUT}/dzero_L{c['layer']}_{tag}.npy", (-xr_raw).astype(np.float32))
            fvu = float((d**2).sum()/max(((xr_raw-xr_raw.mean(0))**2).sum(),1e-9))
            print(f"{name} {tag}: raw-space FVU {fvu:.3f}  |d| {np.abs(d).mean():.4f}", flush=True)
    print("DONE", flush=True)
if __name__ == "__main__":
    main()
