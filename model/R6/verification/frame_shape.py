"""Extract fused printed chassis for the ideal-joint stiffness calculation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_cad import App,HERE,merge
doc=App.openDocument(str(HERE/'R6_0deg.FCStd'))
names=['LowerMount','RailL','RailR','BatteryTray','TowerL','TowerR','Footplate','Knee']
s=merge(*(doc.getObject(k).Shape for k in names))
if len(s.Solids)!=1 or not s.isValid():raise ValueError('The structural chassis is disconnected')
s.exportBrep(str(HERE/'verification/frame.brep'))

for name in ('LowerMount','seat'):
    doc.getObject(name).Shape.exportBrep(str(HERE/'verification'/(name+'.brep')))
