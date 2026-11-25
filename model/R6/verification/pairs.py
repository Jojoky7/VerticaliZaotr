"""OCC Boolean test of every component pair. Fixed relative poses only need testing once."""
import sys, json, argparse, itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_cad import App, placed
cli=argparse.ArgumentParser()
cli.add_argument('cad',type=Path)
cli.add_argument('output',type=Path)
cli.add_argument('--angles',type=int,nargs='+',default=[0,15,30,45,60,75])
args=cli.parse_args()
doc=App.openDocument(str(args.cad.resolve()))
meta=json.loads((args.cad.parent/'assets/assembly.json').read_text())['parts']
shapes={k:doc.getObject(k).Shape for k in meta}
def motion(name):
    g=meta[name]['group']
    if g.startswith(('drive','front','fork')): return 'frame'
    if g in ('driven','idler','shaft','pinion'): return 'housing'
    return g
found={}
checked=0
for angle in args.angles:
    posed={k:placed(s,meta[k]['group'],angle) for k,s in shapes.items()}
    bounds={k:s.BoundBox for k,s in posed.items()}
    for a,b in itertools.combinations(posed,2):
        if angle and motion(a)==motion(b):continue
        x,y=bounds[a],bounds[b]
        if any(min(getattr(x,i+'Max'),getattr(y,i+'Max'))-max(getattr(x,i+'Min'),getattr(y,i+'Min'))<1e-5 for i in 'XYZ'):continue
        checked+=1
        volume=posed[a].common(posed[b]).Volume
        if volume>.25:
            key='/'.join(sorted((a,b)))
            if volume>found.get(key,{}).get('mm3',0):
                found[key]={'mm3':round(volume,3),'deg':angle}
    args.output.write_text(json.dumps(found,indent=2))
    print(angle,len(found),flush=True)

args.output.write_text(json.dumps(dict(components=len(meta),pairs_per_full_pose=len(meta)*(len(meta)-1)//2,angles_deg=args.angles,exact_boolean_tests=checked,threshold_mm3=.25,interferences=found),indent=2))
if found: sys.exit(1)
