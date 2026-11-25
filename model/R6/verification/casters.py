"""Test free roll with conservative Ø128 cylinders, and all external solids during caster yaw."""
import sys,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_cad import App,HERE,Part,V,Y,placed
doc=App.openDocument(str(HERE/'R6_0deg.FCStd'))
meta=json.loads((HERE/'assets/assembly.json').read_text())['parts']
shapes={k:doc.getObject(k).Shape for k in meta}
hits={}
tests=0
def broad(a,b):
    return all(min(getattr(a,k+'Max'),getattr(b,k+'Max'))-max(getattr(a,k+'Min'),getattr(b,k+'Min'))>1e-5 for k in 'XYZ')
def test(a,b,aa,bb,yaw,tilt):
    global tests
    if not broad(a.BoundBox,b.BoundBox):return
    tests+=1
    v=a.common(b).Volume
    if v>.25:
        label=aa+'/'+bb
        if v>hits.get(label,{}).get('mm3',0):hits[label]=dict(mm3=round(v,3),yaw=yaw,seat_deg=tilt)
def external(group):
    return [(name,shapes[name]) for name in meta if meta[name]['group']==group]
for side,sg in [('L',1),('R',-1)]:
    pivot=V(251,100*sg,149)
    # Wheel and hub outside radius 8.1, axle and its retaining nut are separate.
    # This annulus bounds every angular position of all spokes (and the non-cylindrical TPU).
    front=Part.makeCylinder(64,34,V(239,100*sg-17,64),Y)
    front=front.cut(Part.makeCylinder(8.1,36,V(239,100*sg-18,64),Y))
    moving=[('FrontRoll'+side,front),*external('fork'+side),('CasterHub'+side,shapes['CasterHub'+side])]
    for name,one in external('fork'+side):
        test(front,one,'FrontRoll'+side,name,0,0)
    # Opposite front wheel swept radius is bounded by 12 + hypot(64,24) < 81 mm.
    # The two vertical yaw axes are 200 mm apart; opposite assemblies cannot meet.
    floor_bound=Part.makeCylinder(83,169,V(251,100*sg,0)).BoundBox
    targets=[]
    for k in meta:
        g=meta[k]['group']
        if g.startswith(('front','fork')):continue
        for tilt in ([0] if g=='frame' or g.startswith('drive') else range(0,76,5)):
            shape=placed(shapes[k],g,tilt)
            if broad(shape.BoundBox,floor_bound):targets.append((k,shape,tilt))
    for yaw in range(0,360,5):
        for name,one in moving:
            if yaw and name.startswith(('Kingpin','KingWasher','KingSpacer')):continue
            turned=one.copy();turned.rotate(pivot,V(0,0,1),yaw)
            for k,other,tilt in targets:
                test(turned,other,name,k,yaw,tilt)
    # Solid rear running envelope outside central hub, including inner-tread clearance.
    center=V(0,105*sg,64)
    for lo,hi,r in [(88,88.8,18),(88.8,122,64)]:
        w=Part.makeCylinder(r,hi-lo,V(0,min(lo*sg,hi*sg),64),Y)
        w=w.cut(Part.makeCylinder(9,hi-lo+2,V(0,min(lo*sg,hi*sg)-1,64),Y))
        for k in meta:
            g=meta[k]['group']
            if g.startswith(('fork','front','drive')):continue
            for tilt in ([0] if g=='frame' else range(0,76,5)):
                other=placed(shapes[k],g,tilt)
                test(w,other,'RearRoll'+side,k,0,tilt)
result=dict(yaw_step_deg=5,seat_step_deg=5,exact_tests=tests,threshold_mm3=.25,envelope_hits=hits)
(HERE/'verification/caster_sweep.json').write_text(json.dumps(result,indent=2))
print(result)

if hits:sys.exit(1)
