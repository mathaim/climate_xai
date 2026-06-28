"""For each matry_L8 ENSO concept: nested group (G0..G4), fire rate, its best CORE parent
(if it is a child) and its top OUTER children (if a parent), from the co-firing matrix."""
import numpy as np
COF = "/scratch/euh7ys/climate_xai/cofire/cofire_matry_L8.npz"
ENSO = [(1314,"EAus ElNino"),(2532,"EAus LaNina"),(186,"WNAm ElNino"),(1171,"WNAm LaNina"),
        (3919,"WSAm ElNino"),(860,"WSAm LaNina"),(1043,"SH LaNina")]
KNOWN = {99:"global intensity",108:"NH intensity",176:"SH intensity",96:"presence",
         3153:"EAus extreme child",3483:"NH child"}
def grp(i): return "G0" if i<256 else "G1" if i<512 else "G2" if i<1024 else "G3" if i<2048 else "G4"
def tag(i): return KNOWN.get(i,"ENSO" if i in {c for c,_ in ENSO} else "")
def main():
    d=np.load(COF); C=d["cofire"].astype(float); f=d["fire"].astype(float); N=float(d["nodes"][0])
    core=np.arange(512); outer=np.arange(2048,4096)
    for c,lab in ENSO:
        g=grp(c); fr=f[c]/N*100
        fwd=C[core,c]/max(f[c],1); k=int(np.argmax(fwd)); p=core[k]
        Pp=fwd[k]; Pr=C[p,c]/max(f[p],1)
        Pc=C[c,outer]/np.maximum(f[outer],1); kids=outer[np.argsort(Pc)[::-1][:3]]
        print(f"\nc{c} [{lab}]  group={g}  fires {fr:.2f}%")
        if Pp>0.5: print(f"   best core PARENT: c{p} ({grp(p)} {tag(p)})  P(parent|this)={Pp:.2f} P(this|parent)={Pr:.2f}")
        else:      print(f"   no strong core parent (max P(parent|this)={Pp:.2f}) -> not a clear child")
        kk=[(int(j),round(float(C[c,j]/max(f[j],1)),2)) for j in kids if C[c,j]/max(f[j],1)>0.5]
        print(f"   top outer CHILDREN (P(this|child)): {kk if kk else 'none > 0.5 -> not a parent of outer concepts'}")
    print("\nDONE")
if __name__=="__main__": main()
