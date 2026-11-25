"""OCC volume tests for the actuator sweep, caster sweep and actuator internal working stroke."""
import argparse
import json
from pathlib import Path
from build_cad import App, HERE, Part, E, F, U, V, Y, placed, pose

parser=argparse.ArgumentParser()
parser.add_argument('--step',type=int,default=5)
parser.add_argument('--output',type=Path,default=HERE/'fit.json')
args=parser.parse_args()
meta=json.loads((HERE/'assets/assembly.json').read_text())['parts']
doc=App.openDocument(str(HERE/'R5_0deg.FCStd'))
shapes={k:doc.getObject(k).Shape.copy() for k in meta}
def xyz(b):return [b.XMin,b.XMax,b.YMin,b.YMax,b.ZMin,b.ZMax]
def overlaps(a,b):
    av,bv=xyz(a.BoundBox),xyz(b.BoundBox)
    return all(min(av[i+1],bv[i+1])-max(av[i],bv[i])>1e-4 for i in (0,2,4))
def check(a,b,test,at):
    if not overlaps(a,b):return
    solid=a.common(b)
    volume=solid.Volume
    if volume > .25:
        key=' / '.join(test)
        hit=issues.get(key)
        if hit is None or volume>hit['max_mm3']:issues[key]=dict(at_deg=at,max_mm3=round(volume,2),box_mm=[round(v,2) for v in xyz(solid.BoundBox)])

issues={}
clearances={}
def gap(a,b,left,right,angle):
    key=left+' / '+right
    length=float(a.distToShape(b)[0])
    if key not in clearances or length < clearances[key]['gap_mm']:
        clearances[key]=dict(gap_mm=round(length,4),at_deg=angle)
bb=App.BoundBox()
for k,s in shapes.items():bb.add(s.BoundBox)
fixed=[k for k,d in meta.items() if d['group'] in ('frame','seat','back','link') and not k.startswith(('HW_','SP_'))]
case=['Gruzdy0','Gruzdy1','Gruzdy7','Stepper','RearAdapter','OutputRod','RodEye']
for angle in sorted(set(range(0,76,args.step))|{75}):
    posed={k:placed(shapes[k],meta[k]['group'],angle) for k in fixed+case}
    for n in case:
        for k in fixed:
            check(posed[n],posed[k],(n,k),angle)
    for member in ('seat','back','link_left','link_right','UpperMount'):
        for base in ('RailL','RailR','TowerL','TowerR','Footplate','Battery','Knee'):
            check(posed[member],posed[base],(member,base),angle)
    for left,right in [('Gruzdy7','seat'),('Gruzdy0','CanL'),('Gruzdy0','CanR'),('RodEye','UpperMount')]:
        gap(posed[left],posed[right],left,right,angle)
    if angle in (0,75):print('checked lift',angle,flush=True)
# Caster envelope. Wheel profile bounds both the detailed rim and tread on its axle.
# The solid test cylinder deliberately includes the spokes and hub voids.
for sign,side in ((1,'L'),(-1,'R')):
    pivot=V(251,sign*100,149)
    wheel=Part.makeCylinder(64,34,V(239,sign*100-17,64),Y)
    for deg in range(0,360,max(5,args.step)):
        for key,s in [('WheelSweep'+side,wheel),('CasterFork'+side,shapes['CasterFork'+side])]:
            body=s.copy();body.rotate(pivot,V(0,0,1),deg)
            for n in ['Footplate','Tower'+side,'Rail'+side,'Knee','link_left' if sign==1 else 'link_right']:
                check(body,shapes[n],(key,n),deg)
# The slider moves along the Ø10 shaft. Test positive volumes with the housing, excluding the engaged screw.
for k in ('LeadscrewNut','RodBush'):
    for n in ('Gruzdy4','Gruzdy6','OutputRod'):
        check(shapes[k],shapes[n],(k,n),0)
for t in (0,.25,.5,.75,1):
    s=shapes['Gruzdy4'].copy();s.translate(U*(t*pose(75)[1]))
    for n in ('Gruzdy0','Gruzdy1','Gruzdy7'):
        check(s,shapes[n],('carriage',n),round(75*t,1))
for side in ('L','R'):
    for a,b in (('HubNut','Rim0'),('HubScrew','Rim0'),('Dshaft','Rim0'),('Dshaft','HubScrew'),('HubNut','HubScrew'),('CasterHub','Rim280')):
        check(shapes[a+side],shapes[b+side],(a+side,b+side),0)
raised=App.BoundBox()
for name,shape in shapes.items():raised.add(placed(shape,meta[name]['group'],75).BoundBox)
gap(shapes['Gruzdy4'],shapes['Gruzdy1'],'carriage','housing',0)
_,stroke,_=pose(75)
result=dict(closed_bounds_mm=xyz(bb),size_mm=[bb.XLength,bb.YLength,bb.ZLength],raised_size_mm=[raised.XLength,raised.YLength,raised.ZLength],required_stroke_mm=stroke,
    carriage_axial_mm=[[32,47.4],[32+stroke,47.4+stroke]],shaft_cylindrical_span_mm=[17.2,179.3],
    step_deg=args.step,caster_step_deg=max(5,args.step),sampled_clearances=clearances,collision_volume_tolerance_mm3=.25,interferences=issues)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
