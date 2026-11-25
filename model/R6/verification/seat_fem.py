"""Reproduce the seated, 8 kg comparison. Units are N, mm and MPa."""
import argparse
import json
import math
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def tool(name):
    return shutil.which(name) or str(Path("/Applications/FreeCAD.app/Contents/Resources/bin") / name)


def mesh(case, size, work):
    geo = work / "seat.geo"
    geo.write_text(
        f'Merge "{HERE / (case + ".brep")}";\n'
        f'Mesh.MeshSizeMin={size/2}; Mesh.MeshSizeMax={size}; '
        'Mesh.MeshSizeFromCurvature=10; Mesh.ElementOrder=2; Mesh.HighOrderOptimize=1;\n'
    )
    subprocess.run([tool("gmsh"), "-3", str(geo), "-format", "inp", "-o", str(work/"mesh.inp"),
                    "-v", "2", "-nt", "2"], check=True)


def parse_mesh(path):
    nodes, tets, weights = {}, [], defaultdict(float)
    section = ""
    for line in path.read_text().splitlines(keepends=True):
        if line.startswith("*"):
            key = line.strip().upper()
            section = "nodes" if key == "*NODE" else ""
            if "TYPE=C3D10," in key:
                section = "tetrahedra"
            if "TYPE=CPS6," in key:
                section = "triangles"
            continue
        if not section:
            continue
        fields = line.strip().split(",")
        if section == "nodes":
            nodes[int(fields[0])] = tuple(map(float, fields[1:4]))
        elif section == "tetrahedra":
            tets.append(line)
        else:
            tri = list(map(int, fields[1:7]))
            p, q, r = (nodes[i] for i in tri[:3])
            cx, cy = ((p[j]+q[j]+r[j])/3 for j in (0, 1))
            if all(abs(nodes[i][2]-248.5) < .001 for i in tri) and 100 < cx < 212 and abs(cy) < 55:
                area = abs((q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0]))/2
                # Consistent nodal loads for a quadratic triangle with uniform pressure.
                for i in tri[3:]:
                    weights[i] += area/3
    return nodes, tets, weights


def nodeset(file, label, ids):
    file.write(f"*NSET,NSET={label}\n")
    for j in range(0, len(ids), 12):
        file.write(",".join(map(str, ids[j:j+12]))+"\n")


def solve(work):
    nodes, tets, weights = parse_mesh(work/"mesh.inp")
    pivot = [i for i,(x,y,z) in nodes.items()
             if abs(math.hypot(x-280,z-260)-2.65)<.025 and 90.99<=abs(y)<=97.01]
    flange = [i for i,(x,y,z) in nodes.items() if 243.99<=z<=248.51 and any(
        abs(math.hypot(x-h,y-k)-1.7)<.025 for h in (173,191) for k in (-49,-15))]
    if not weights or len(pivot)<12 or len(flange)<12:
        raise ValueError("Support or load selection is empty for this geometry")
    area = sum(weights.values())
    with (work/"load.inp").open("w") as f:
        f.write("*NODE\n")
        for i,(x,y,z) in nodes.items():
            f.write(f"{i},{x:.8f},{y:.8f},{z:.8f}\n")
        f.write("*ELEMENT,TYPE=C3D10,ELSET=seat\n")
        f.writelines(tets)
        nodeset(f,"pin",pivot)
        nodeset(f,"flange",flange)
        nodeset(f,"monitor",list(weights))
        f.write("*MATERIAL,NAME=surrogate\n*ELASTIC\n1800,.35\n*SOLID SECTION,ELSET=seat,MATERIAL=surrogate\n"
                "*TRANSFORM,NSET=flange,TYPE=R\n137,0,192,0,1,0\n*BOUNDARY\npin,1,1\npin,3,3\n")
        f.write(f"{pivot[0]},2,2\nflange,1,1\n*STEP\n*STATIC\n*CLOAD\n")
        for i, value in weights.items():
            f.write(f"{i},3,{-8*9.80665*value/area:.10f}\n")
        f.write("*NODE PRINT,NSET=monitor\nU\n*END STEP\n")
    with (work/"ccx.log").open("w") as log:
        subprocess.run([tool("ccx"),"-i","load"],cwd=work,stdout=log,stderr=log,
                       env={**os.environ,"OMP_NUM_THREADS":"2"},check=True)
    rows = (work/"load.dat").read_text().splitlines()
    uz = [float(r.split()[-1]) for r in rows if len(r.split())==4 and r.split()[0].isdigit()]
    return dict(elements=len(tets),area_mm2=round(area,1),peak_loaded_patch_down_mm=-min(uz),
                flange_nodes=len(flange),pin_nodes=len(pivot))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("case", choices=["seat"])
    cli.add_argument("--size",type=int,choices=[4,6],default=4)
    cli.add_argument("--work",type=Path,required=True)
    args = cli.parse_args()
    dest = args.work.resolve()/f"{args.case}_{args.size}"
    dest.mkdir(parents=True,exist_ok=True)
    mesh(args.case,args.size,dest)
    result = solve(dest)
    (dest/"report.json").write_text(json.dumps(result, indent=2)+"\n")
    print(result)
