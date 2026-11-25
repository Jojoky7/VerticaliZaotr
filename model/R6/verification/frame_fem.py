"""Ideal bonded joints, full printed chassis, near-start 8 kg interface loads from MuJoCo."""
import argparse, json, os, subprocess
from pathlib import Path
import numpy as np
from mount_fem import binary,write_set,HERE
p=argparse.ArgumentParser()
p.add_argument('work',type=Path)
p.add_argument('--size',type=float,default=6)
p.add_argument('--report',type=Path,default=HERE/'frame_static.json')
p.add_argument('--loads',type=Path,default=HERE/'frame_loads.json')
p.add_argument('--reuse-mesh', action='store_true', help='Use the existing mesh in the specified work directory')
args=p.parse_args()
w=args.work.resolve();w.mkdir(parents=True,exist_ok=True)
geo=w/'frame.geo'
geo.write_text(f'Merge "{HERE}/frame.brep";\nMesh.MeshSizeMin={args.size/3}; Mesh.MeshSizeMax={args.size}; Mesh.MeshSizeFromCurvature=10; Mesh.ElementOrder=2; Mesh.HighOrderOptimize=1;\n')
if not args.reuse_mesh:
  with (w/'mesh.log').open('w') as log:
    subprocess.run([binary('gmsh'),'-3',str(geo),'-format','inp','-o',str(w/'mesh.inp'),'-nt','2'],stdout=log,stderr=log,check=True)
nodes={};tets=[];section=''
for line in (w/'mesh.inp').read_text().splitlines(keepends=True):
    if line.startswith('*'):section=line.upper().strip();continue
    if section=='*NODE':
        i,*xyz=line.split(',');nodes[int(i)]=np.array(list(map(float,xyz[:3])))
    if 'TYPE=C3D10,' in section:tets.append(line)
def select(predicate):
    ids=[i for i,(x,y,z) in nodes.items() if predicate(x,y,z)]
    if len(ids)<3:raise ValueError('Missing load or support region')
    return ids
def radial(x,z,cx,cz,r):return abs(np.hypot(x-cx,z-cz)-r)<.02
rear=select(lambda x,y,z:60.99<=abs(y)<=65.01 and any(radial(x,z,xx,zz,1.65) for xx in (-4,29) for zz in (55,73)))
caster=select(lambda x,y,z:148.99<=z<=159.01 and radial(x,abs(y),251,100,3.15))
regions={'housing':select(lambda x,y,z: abs(y+32)<14 and radial(x,z,45,40,3.15)),
         'seat':select(lambda x,y,z:64.99<=abs(y)<=75.01 and radial(x,z,280,260,4.3)),
         'link':select(lambda x,y,z:64.99<=abs(y)<=75.01 and radial(x,z,280,160,4.3))}
wrenches=json.loads(args.loads.read_text())
loads={i:np.zeros(3) for i in nodes}
def apply(region,origin,force,moment):
    # Minimum norm nodal force distribution preserving all six resultant components.
    blocks=[]
    for i in region:
        x,y,z=nodes[i]-origin
        skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        blocks.append(np.vstack((np.eye(3),skew)))
    matrix=np.concatenate(blocks,axis=1)
    target=np.r_[force,moment]
    f=np.linalg.lstsq(matrix,target,rcond=None)[0]
    if max(abs(matrix@f-target))>1e-5:raise ValueError('Unbalanced input wrench')
    for i,v in zip(region,f.reshape(-1,3)):loads[i]+=v
for k,origin in [('housing',[45,-32,40]),('seat',[280,0,260]),('link',[280,0,160])]:
    v=wrenches[k]
    apply(regions[k],origin,v['force_N'],np.array(v['moment_Nm'])*1000)
battery=select(lambda x,y,z:abs(z-105)<.001 and -53<x<12 and abs(y)<75.5)
feet=select(lambda x,y,z:abs(z-127)<.001 and 300<x<342 and abs(y)<58)
apply(battery,[-20.5,0,105],[0,0,-2.2*9.80665],[0,0,0])
apply(feet,[320,0,127],[0,0,-8*.136*9.80665],[0,0,0])
with (w/'frame.inp').open('w') as f:
    f.write('*NODE\n')
    for i,(x,y,z) in nodes.items():f.write(f'{i},{x:.8f},{y:.8f},{z:.8f}\n')
    f.write('*ELEMENT,TYPE=C3D10,ELSET=frame\n');f.writelines(tets)
    write_set(f,'rear',rear);write_set(f,'caster',caster)
    write_set(f,'ymount',[i for i in rear if nodes[i][1]<0])
    write_set(f,'allnodes',list(nodes))
    f.write('*MATERIAL,NAME=surrogate\n*ELASTIC\n1800,.35\n*DENSITY\n6.5e-10\n*SOLID SECTION,ELSET=frame,MATERIAL=surrogate\n'
            '*BOUNDARY\nrear,1,1\nrear,3,3\nymount,2,2\ncaster,3,3\n*STEP\n*STATIC\n*DLOAD\nframe,GRAV,9806.65,0,0,-1\n*CLOAD\n')
    for i,force in loads.items():
        for j,v in enumerate(force):
            if abs(v)>1e-10:f.write(f'{i},{j+1},{v:.10g}\n')
    f.write('*NODE PRINT,NSET=allnodes\nU\n*EL PRINT,ELSET=frame\nS\n*END STEP\n')
with (w/'ccx.log').open('w') as log:
    subprocess.run([binary('ccx'),'-i','frame'],cwd=w,stdout=log,stderr=log,env={**os.environ,'OMP_NUM_THREADS':'2'},check=True)
disps={};stress=[]
for line in (w/'frame.dat').read_text().splitlines():
    row=line.split()
    if not row or not row[0].isdigit():continue
    if len(row)==4:disps[int(row[0])]=float(np.linalg.norm(np.array(row[1:],float)))
    if len(row)==8:
        sx,sy,sz,xy,xz,yz=map(float,row[2:]);stress.append(np.sqrt(((sx-sy)**2+(sy-sz)**2+(sz-sx)**2)/2+3*(xy*xy+xz*xz+yz*yz)))
peak=max(disps,key=disps.get)
report=dict(elements=len(tets),mesh_mm=args.size,peak_displacement_mm=disps[peak],peak_location_mm=nodes[peak].tolist(),
            region_peak_mm={k:max(disps[i] for i in reg) for k,reg in regions.items()},max_vm_MPa=max(stress),
            seat_deg=wrenches['seat_deg'],near_start_load_kg=8)
args.report.write_text(json.dumps(report,indent=2));print(report)
