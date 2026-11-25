"""Linear static bearing-load test of the printed lower bracket. Units: N, mm, MPa."""
import argparse, collections, json, math, os, shutil, subprocess
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def binary(name):
    return shutil.which(name) or str(Path('/Applications/FreeCAD.app/Contents/Resources/bin')/name)
def grok_mesh(path):
    nodes={}; tetra=[]; faces=[]
    section=''
    for line in path.read_text().splitlines(keepends=True):
        if line.startswith('*'):
            section=line.upper().strip()
            continue
        if section=='*NODE':
            i,*coords=line.split(',')
            nodes[int(i)]=np.array(list(map(float,coords[:3])))
        elif 'TYPE=C3D10,' in section:
            tetra.append(line)
        elif 'TYPE=CPS6,' in section:
            ids=list(map(int,line.split(',')[1:7]))
            if all(abs(math.hypot(nodes[i][0]-45,nodes[i][2]-40)-3.15)<.02 for i in ids):
                faces.append(ids)
    return nodes,tetra,faces
def write_set(f,name,ids):
    f.write('*NSET,NSET='+name+'\n')
    for i in range(0,len(ids),12):f.write(','.join(map(str,ids[i:i+12]))+'\n')
def solve(work,h,angles):
    work.mkdir(exist_ok=True,parents=True)
    geo=work/'bracket.geo'
    geo.write_text(f'Merge "{HERE}/LowerMount.brep";\nMesh.MeshSizeMin={h/2}; Mesh.MeshSizeMax={h}; Mesh.MeshSizeFromCurvature=15; Mesh.ElementOrder=2; Mesh.HighOrderOptimize=1;\n')
    subprocess.run([binary('gmsh'),'-3',str(geo),'-order','2','-clmax',str(h),'-clmin',str(h/2),'-optimize_ho','-format','inp','-o',str(work/'mesh.inp'),'-v','1','-nt','2'],check=True)
    nodes,tets,faces=grok_mesh(work/'mesh.inp')
    pads=[i for i,(x,y,z) in nodes.items() if abs(z-46)<.001 and 65<=abs(y)<=75 and 38<=x<=77 and not any(k-.9<=x<=k+5.9 and 65.4<=abs(y)<=74.6 for k in (50,69))]
    keys=[i for i,(x,y,z) in nodes.items() if min(abs(x-50),abs(x-69))<.001 and 41.79<=z<=45.25 and 66.29<=abs(y)<=73.71]
    lateral=[i for i,(x,y,z) in nodes.items() if abs(y-66.3)<.001 and 41.79<=z<=45.25 and 50<=x<=55]
    assert min(len(pads),len(keys),len(lateral),len(faces))>=3,(len(pads),len(keys),len(lateral),len(faces))
    results=[]
    for deg in angles:
        a=math.radians(deg)
        x=280-98*math.cos(a)-28*math.sin(a)
        z=260+98*math.sin(a)-28*math.cos(a)
        axis=np.array([45-x,0,40-z])
        axis/=np.linalg.norm(axis)
        weights=collections.defaultdict(float)
        for face in faces:
            p,q,r=[nodes[i] for i in face[:3]]
            center=(p+q+r)/3
            radial=np.array([center[0]-45,0,center[2]-40])
            cosine=np.dot(radial,axis)/np.linalg.norm(radial)
            if cosine>0:
                area=np.linalg.norm(np.cross(q-p,r-p))/2
                for i in face[3:]: weights[i]+=area*cosine/3
        total=sum(weights.values())
        name='angle'+str(deg)
        with (work/(name+'.inp')).open('w') as f:
            f.write('*NODE\n')
            for i,(x,y,z) in nodes.items():f.write(f'{i},{x:.8f},{y:.8f},{z:.8f}\n')
            f.write('*ELEMENT,TYPE=C3D10,ELSET=body\n');f.writelines(tets)
            for label,ids in [('pads',pads),('keys',keys),('lateral',lateral),('bore',list(weights))]:write_set(f,label,ids)
            f.write('*MATERIAL,NAME=printed\n*ELASTIC\n1800,.35\n*SOLID SECTION,ELSET=body,MATERIAL=printed\n'
                    '*BOUNDARY\npads,3,3\nkeys,1,1\nlateral,2,2\n*STEP\n*STATIC\n*CLOAD\n')
            for i,w in weights.items():
                for j in (0,2):f.write(f'{i},{j+1},{335.103216*w/total*axis[j]:.9f}\n')
            f.write('*NODE PRINT,NSET=bore\nU\n*EL PRINT,ELSET=body\nS\n*END STEP\n')
        with (work/(name+'.log')).open('w') as log:
            subprocess.run([binary('ccx'),'-i',name],cwd=work,stdout=log,stderr=log,check=True,env={**os.environ,'OMP_NUM_THREADS':'2'})
        disps=[]; stress=[]
        for ln in (work/(name+'.dat')).read_text().splitlines():
            row=ln.split()
            if not row or not row[0].isdigit():continue
            if len(row)==4:disps.append(np.linalg.norm(np.array(row[1:],float)))
            if len(row)==8:
                sx,sy,sz,xy,xz,yz=map(float,row[2:])
                stress.append(math.sqrt(((sx-sy)**2+(sy-sz)**2+(sz-sx)**2)/2+3*(xy*xy+xz*xz+yz*yz)))
        results.append(dict(seat_deg=deg,force_N=335.103216,max_pin_disp_mm=max(disps),
                            peak_vm_MPa=max(stress),integration_point_p99_vm_MPa=float(np.percentile(stress,99)),
                            elements=len(tets),mesh_mm=h))
    return results
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('report',type=Path);p.add_argument('--size',type=float,default=2.5);p.add_argument('--angles',type=int,nargs='+',default=[0,38,75]);args=p.parse_args()
    results=solve(args.work.resolve(),args.size,args.angles)
    args.report.write_text(json.dumps(results,indent=2));print(results)
