"""G1 (256-511) concepts ranked by intensity-relatedness to parent 99, with specialization fingerprints."""
import numpy as np, pandas as pd
from src.analysis.ar_intensity.regions import REGIONS
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; THRESH = 0.1; AR_START = pd.Timestamp("1979-01-01")
REGN = list(REGIONS); MON = "J F M A M J J A S O N D".split(); G1 = (256, 512)
def main():
    fire = None; regcnt = np.zeros((4, 4096)); mocnt = np.zeros((12, 4096)); ivtc = np.zeros(4096); nr = 0
    for ri, r in enumerate(REGN):
        t = np.load(f"{TRACK}/track_matry_{r}.npz"); A = t["A_max"].astype(np.float32); ivt = t["ivt"].astype(float); ti = t["tindex"]
        f = A > THRESH
        fire = f.copy() if fire is None else (fire | f)
        regcnt[ri] = f.sum(0)
        mo = (AR_START + pd.to_timedelta(6 * (ti - 1), unit="h")).month.values
        for mm in range(1, 13): mocnt[mm - 1] += f[mo == mm].sum(0)
        ok = np.isfinite(ivt); Ao = A[ok]
        iz = (ivt[ok] - ivt[ok].mean()) / (ivt[ok].std() + 1e-9)
        ivtc += (Ao * iz[:, None]).mean(0) / (Ao.std(0) + 1e-9); nr += 1
        del A, f, t
    ivtc /= nr; rate = fire.mean(0); p99 = fire[:, 99]
    def conc(c): tot = c.sum(0); fr = c / np.maximum(tot, 1); return 1 - (-(fr*np.log(fr+1e-12)).sum(0)/np.log(c.shape[0]))
    rcn, mcn = conc(regcnt), conc(mocnt); domreg = np.argmax(regcnt, 0); dommo = np.argmax(mocnt, 0)
    g1 = np.arange(*G1); both = (fire[:, g1] & p99[:, None]).sum(0).astype(float)
    contain = both / np.maximum(fire[:, g1].sum(0), 1)
    order = g1[np.argsort(-ivtc[g1])]
    print(f"PARENT 99 (G0):  IVTcorr={ivtc[99]:+.2f}  rate={rate[99]*100:.0f}%  spec r/m={rcn[99]:.2f}/{mcn[99]:.2f}  {REGN[domreg[99]][:7]}/{MON[dommo[99]]}\n")
    print(f"{'G1':>5} {'IVTcorr':>7} {'rate':>6} {'specR':>5} {'specM':>5} {'topReg':>8} {'topMo':>5} {'P(99|c)':>7}")
    for c in order[:25]:
        print(f"{c:>5} {ivtc[c]:>+7.2f} {rate[c]*100:>5.1f}% {rcn[c]:>5.2f} {mcn[c]:>5.2f} {REGN[domreg[c]][:7]:>8} {MON[dommo[c]]:>5} {contain[c-256]:>7.2f}")
if __name__ == "__main__":
    main()
