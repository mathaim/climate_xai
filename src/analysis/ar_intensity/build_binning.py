"""Stage 4: per-region 10/40/40/10 intensity bins."""
import json, pandas as pd
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity.binning import assign_bins, BINS
OUTDIR="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
def main():
    df=pd.read_parquet(f"{OUTDIR}/ar_intensity.parquet"); thr={}; parts=[]
    for r in REGIONS:
        s=df[df.region==r].copy()
        p10,p50,p90=s.max_ivt.quantile([.1,.5,.9])
        s["intensity_bin"]=assign_bins(s.max_ivt.values,p10,p50,p90)
        thr[r]={"p10":float(p10),"p50":float(p50),"p90":float(p90)}; parts.append(s)
    out=pd.concat(parts,ignore_index=True)
    out.to_parquet(f"{OUTDIR}/ar_intensity_binned.parquet")
    json.dump(thr,open(f"{OUTDIR}/region_thresholds.json","w"),indent=2)
    print(f"saved {len(out)} rows + region_thresholds.json")
    for r in REGIONS:
        vc=out[out.region==r].intensity_bin.value_counts()
        print(f"  {r}: "+"  ".join(f"{b}={int(vc.get(b,0))}" for b in BINS))
if __name__=="__main__": main()
