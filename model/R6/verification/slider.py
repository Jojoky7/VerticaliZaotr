"""Sweep the entire assembled nut carrier in the fixed housing, in axial coordinates."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_cad import HERE,App,U,pose,moved
d=App.openDocument(str(HERE/'R6_0deg.FCStd'))
meta=json.loads((HERE/'assets/assembly.json').read_text())['parts']
names=[k for k,v in meta.items() if v['group']=='carriage']
fixed=[k for k,v in meta.items() if v['group']=='housing']
travel=pose(75)[1]
hits={};tests=0
def overlap(a,b):
    return all(min(getattr(a,i+'Max'),getattr(b,i+'Max'))-max(getattr(a,i+'Min'),getattr(b,i+'Min'))>1e-5 for i in 'XYZ')
# The axial coordinate is constant across lift after removing barrel pitch.
for dist in [*range(0,122,4),travel]:
    for k in names:
        s=moved(d.getObject(k).Shape,U*dist)
        for f in fixed:
            q=d.getObject(f).Shape
            if not overlap(s.BoundBox,q.BoundBox):continue
            tests+=1
            v=s.common(q).Volume
            if v>.25 and v>hits.get(k+'/'+f,{}).get('mm3',0):hits[k+'/'+f]=dict(mm3=v,extension_mm=dist)
result=dict(step_mm=4,travel_mm=travel,boolean_tests=tests,threshold_mm3=.25,interferences=hits)
(HERE/'verification/slider.json').write_text(json.dumps(result,indent=2))
print(result)
if hits:sys.exit(1)
