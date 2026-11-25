"""Sample one tooth cycle using OCC sections of the installed gear solids (straight teeth)."""
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_cad import App,Part,HERE,V,U,installed
from shapely.geometry import Polygon
from shapely.affinity import rotate
from shapely.ops import unary_union
import numpy as np
cli=argparse.ArgumentParser()
cli.add_argument('cad',type=Path)
cli.add_argument('report',type=Path)
cli.add_argument('--search',action='store_true')
args=cli.parse_args()
doc=App.openDocument(str(args.cad.resolve()))
zero=installed(0,0,0)
lateral=V(-U.z,0,U.x)
centers={2:(-1.0795,-43.91055),3:(-1.0795,-16.93197),5:(-1.0795,-.078015)}
def polygon(index,station):
    shape=doc.getObject('Gruzdy'+str(index)).Shape
    section=shape.slice(U,(zero+U*station).dot(U))
    loops=[]
    for wire in section:
        pts=wire.discretize(Deflection=.006)
        poly=Polygon([((p-zero).dot(lateral),(p-zero).y) for p in pts])
        loops.append(poly.buffer(0))
    return max(loops,key=lambda p:p.area)
def measure(a,b,delta,angles):
    first,second=profiles[a],profiles[b]
    ratio=-4 if b==5 else -1
    return max(rotate(first,t,origin=centers[a]).intersection(
        rotate(second,ratio*t+delta,origin=centers[b])).area for t in angles)
results={}
for station in (8,12,16):
    profiles={k:polygon(k,station) for k in centers}
    for a,b in ((2,3),(3,5)):
        phase=0.
        if args.search:
            coarse=np.linspace(-180/(12 if b==5 else 48),180/(12 if b==5 else 48),41)
            phase=min(coarse,key=lambda d:measure(a,b,d,np.linspace(0,7.5,5)))
        peak=measure(a,b,phase,np.linspace(0,7.5,61))
        results[f'{station}:{a}/{b}']=dict(phase_deg=float(phase),max_area_mm2=round(peak,6))
        print(station,a,b,phase,peak,flush=True)
args.report.write_text(json.dumps(results,indent=2))

if any(v['max_area_mm2']>.001 for v in results.values()) and not args.search:sys.exit(1)
