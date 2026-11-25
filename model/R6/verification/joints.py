"""Test the physical mating faces, paired bores, insertion path and printable exports."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_cad import App,V,HERE,Part,tube,box,moved,placed
import Mesh
doc=App.openDocument(str(HERE/'R6_0deg.FCStd'))
def shape(k):return doc.getObject(k).Shape
issues=[]
def check(state,label):
    if not state:issues.append(label)
report={}
mount=shape('LowerMount')
for name,y in [('RailL',70),('RailR',-70)]:
    rail=shape(name)
    face=Part.makePlane(39,10,V(38,y-5,46))
    area=face.common(mount).common(rail).Area
    check(area>100,name+' bearing face')
    interference=mount.common(rail).Volume
    check(interference<.01,name+' interference')
    keyed={}
    for dx in [-.3,.3]:
        vol=moved(mount,V(dx,0,0)).common(rail).Volume
        keyed[dx]=round(vol,2)
        check(vol>1,name+' shear key stop')
    insertion=[moved(mount,V(0,0,z)).common(rail).Volume for z in range(81)]
    check(max(insertion)<.01,name+' top insertion')
    report[name]=dict(bearing_mm2=area,nominal_overlap_mm3=interference,
                      displaced_overlap_mm3=keyed,max_insertion_overlap_mm3=max(insertion))
for name in ('LowerMount','RailL','RailR','BatteryTray','TowerL','TowerR','Footplate','Knee','seat','UpperMount','CarriageLid','Gruzdy4'):
    check(len(shape(name).Solids)==1 and shape(name).isValid(),name+' topology')
meshes={}
for fold in ['STL','Проба посадки']:
    meshes[fold]={}
    for f in sorted((HERE/fold).glob('*.stl')):
        m=Mesh.Mesh(str(f));passed=m.isSolid() and m.countComponents()==1
        check(passed,f.name+' STL')
        meshes[fold][f.name]=dict(watertight=m.isSolid(),components=m.countComponents())
meta=json.loads((HERE/'assets/assembly.json').read_text())['parts']
origins=json.loads((HERE/'assets/assembly.json').read_text())['origins']
raised=App.openDocument(str(HERE/'R6_75deg.FCStd'))
def extents(b):
    return (b.XMin,b.YMin,b.ZMin,b.XMax,b.YMax,b.ZMax)
mesh_error=raised_error=volume_error=0
for k,part in meta.items():
    g=part['group']
    mesh=Mesh.Mesh(str(HERE/'assets'/(k+'.stl')))
    mesh.translate(*origins[g])
    a,b=extents(mesh.BoundBox),extents(shape(k).optimalBoundingBox(False))
    delta=max(abs(x-y) for x,y in zip(a,b))
    check(delta<.3,k+' CAD to simulation mesh bounds')
    mesh_error=max(mesh_error,delta)
    center=mesh.BoundBox.Center
    mesh.translate(-center.x,-center.y,-center.z)
    vol=abs(mesh.Volume/shape(k).Volume-1)
    check(vol<.025,k+' CAD to mesh volume')
    volume_error=max(volume_error,vol)
    a,b=extents(placed(shape(k),g,75).BoundBox),extents(raised.getObject(k).Shape.BoundBox)
    delta=max(abs(x-y) for x,y in zip(a,b))
    check(delta<.001,k+' raised CAD pose')
    raised_error=max(raised_error,delta)
fasteners={}
for bolt in (k for k in meta if k.endswith('Bolt')):
    nut=bolt[:-4]+'Nut'
    if nut not in meta:
        issues.append(bolt+' missing nut');continue
    clearance=shape(bolt).distToShape(shape(nut))[0]
    check(clearance<.3,bolt+' missing thread engagement')
    fasteners[bolt]=round(clearance,4)
result=dict(rail_tests=report,paired_nuts=fasteners,meshes=meshes,max_asset_bound_error_mm=mesh_error,max_relative_asset_volume_error=volume_error,max_raised_bound_error_mm=raised_error,issues=issues)
(HERE/'verification/joints.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print(report,issues)
if issues:sys.exit(1)
