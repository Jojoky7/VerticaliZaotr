"""Extract interface wrenches in world axes, on the frame, from the 8 kg lift."""
import argparse,json,sys
from pathlib import Path
import mujoco as mj
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stand import HERE,Bench
args=argparse.ArgumentParser()
args.add_argument('--angle',type=float,help='Capture while crossing this angle instead of at 3 seconds')
args.add_argument('--output',type=Path,default=HERE/'verification/frame_loads.json')
opts=args.parse_args()
b=Bench(8,mobile=True)
if opts.angle is None:
    for i in range(round(3/b.model.opt.timestep)):b.step()
else:
    while np.rad2deg(b.q('lift'))<opts.angle and b.d.time<160:b.step()
    if np.rad2deg(b.q('lift'))<opts.angle:raise RuntimeError('Pose unreachable')
mj.mj_forward(b.model,b.d);mj.mj_rnePostConstraint(b.model,b.d)
out={}
for name in ('housing','seat','link'):
    i=b.model.body(name).id
    center=b.d.subtree_com[b.model.body_rootid[i]]
    torque,force=np.split(-b.d.cfrc_int[i],2)
    local=torque-np.cross(b.d.xpos[i]-center,force)
    out[name]=dict(force_N=force.tolist(),moment_Nm=local.tolist(),origin_m=b.d.xpos[i].tolist())
out['seat_deg']=np.rad2deg(b.q('lift'));out['time_s']=b.d.time
out['subtree_weight_N']=9.80665*sum(b.model.body_subtreemass[b.model.body(k).id] for k in ('housing','seat','link'))
print(out)
opts.output.write_text(json.dumps(out,indent=2))
