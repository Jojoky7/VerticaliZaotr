"""Raise, hold, lower and settle each load on four wheels in MuJoCo."""
import json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stand import Bench,HERE
runs=[]
for mass in (5,6,8):
    b=Bench(mass,mobile=True)
    dt=b.model.opt.timestep
    for i in range(round((b.travel/b.speed+2)/dt)):b.step()
    raised=math.degrees(b.q('lift'))
    for i in range(round(5/dt)):b.step(0)
    hold=math.degrees(b.q('lift'))
    for i in range(round((b.travel/b.speed+2)/dt)):b.step(-1)
    for i in range(round(2/dt)):b.step(0)
    lowered=math.degrees(b.q('lift'))
    result=dict(mass_kg=mass,raised_deg=raised,after_5s_deg=hold,lowered_deg=lowered,
                final_extension_mm=1000*b.q('extension'),max_constraint_error_mm=1000*b.max_error,
                numerical_warnings=int(np.sum(b.d.warning.number)))
    result['passed']=bool(raised>74.5 and abs(raised-hold)<.2 and abs(lowered)<.3 and
                          abs(result['final_extension_mm'])<.5 and b.max_error<.0004 and not result['numerical_warnings'])
    runs.append(result);print(result,flush=True)
(HERE/'verification/cycles.json').write_text(json.dumps(runs,indent=2))
if not all(x['passed'] for x in runs):sys.exit(1)
