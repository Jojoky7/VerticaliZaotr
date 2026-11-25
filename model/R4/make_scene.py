"""Generate the MuJoCo mechanism from the same part placements as the FreeCAD assembly."""
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np

HERE=Path(__file__).resolve().parent
spec=json.loads((HERE/'assets/assembly.json').read_text())
origin={name:np.array(v)*.001 for name,v in spec['origins'].items()}
axis=spec['direction']
def add(parent,tag,**kw):
    return ET.SubElement(parent,tag,{k:(' '.join(f'{x:.12g}' for x in v) if isinstance(v,(tuple,list,np.ndarray)) else str(v)) for k,v in kw.items()})
root=ET.Element('mujoco',model='R4 / central Gruzdy actuator')
add(root,'compiler',angle='radian',meshdir='assets',balanceinertia='true')
add(root,'option',gravity=(0,0,-9.80665),timestep=.002,integrator='implicitfast',iterations=70,tolerance=1e-10)
add(root,'statistic',center=(.14,0,.31),extent=.75)
vis=add(root,'visual');add(vis,'global',offwidth=1120,offheight=720);add(vis,'headlight',ambient=(.42,.42,.42),diffuse=(.68,.68,.68))
defaults=add(root,'default');add(defaults,'geom',contype=0,conaffinity=0,friction=(.8,.002,.0001));add(defaults,'joint',armature=1e-6)
assets=add(root,'asset');add(assets,'texture',type='skybox',builtin='gradient',rgb1=(.34,.44,.50),rgb2=(.82,.88,.89),width=256,height=1536)
w=add(root,'worldbody');add(w,'light',pos=(.3,-.5,1.3),dir=(-.2,.3,-1),directional='true')
add(w,'geom',name='ground',type='plane',size=(0,0,.03),rgba=(.62,.68,.69,1),priority=1,solref=(.006,1),solimp=(.98,.995,.001),contype=1,conaffinity=1)
bodies={'frame':add(w,'body',name='frame',pos=(0,0,0))}
def body(name,parent,joint=None,ja=(0,1,0),**attr):
    b=add(bodies[parent],'body',name=name,pos=origin[name]-origin[parent])
    bodies[name]=b
    if joint:add(b,'joint',name=joint,axis=ja,**attr)
    return b
body('seat','frame','lift',range=(0,math.radians(75)),limited='true',damping=.10)
body('back','seat','back_hinge')
body('link','frame','lower_hinge')
body('housing','frame','barrel_pitch',damping=.007)
body('carriage','housing','extension',ja=axis,type='slide',range=(0,.128),limited='true',armature=12,damping=1.8)
for name in ('shaft','pinion','idler','driven'):
    body(name,'housing',name+'_angle',ja=axis,damping=.0002)
for side in ('L','R'):
    body('drive'+side,'frame','wheel'+side,damping=.00016,armature=.001)
    body('fork'+side,'frame','yaw'+side,ja=(0,0,1),damping=.0016)
    body('front'+side,'fork'+side,'caster'+side,damping=.00015)
    for group in ['drive'+side,'front'+side]:
        add(bodies[group],'geom',name='contact_'+group,type='ellipsoid' if group.startswith('front') else 'cylinder',
            size=(.064,.064,.056) if group.startswith('front') else (.064,.017),quat=(.70710678,.70710678,0,0),
            mass=.0001,rgba=(0,0,0,0),group=3,condim=3)
for name,meta in spec['parts'].items():
    add(assets,'mesh',name=name,file=name+'.stl',scale=(.001,.001,.001))
    color=[c/255 for c in meta['color']]
    alpha=.35 if name in ('Gruzdy0','Gruzdy1','Gruzdy7') else 1
    add(bodies[meta['group']],'geom',name=name,type='mesh',mesh=name,rgba=(*color,alpha),mass=meta['mass'])
F=np.array(spec['F'])*.001
for b,n in [('seat','rocker_pin'),('carriage','link_tip')]:
    add(bodies[b],'site',name=n,pos=F-origin[b],size=.002,rgba=(.91,.58,.20,1))
hip=np.array([.136,0,.299]);origin['man_trunk']=hip
trunk=body('man_trunk','seat','hip_compensation')
fabric=(.16,.51,.53,1);pants=(.16,.25,.36,1);skin=(.84,.62,.48,1);shoe=(.20,.22,.24,1)
def person(b,name,shape,mass,rgba,**kw):
    add(b,'geom',name=name,type=shape,mass=6*mass,rgba=rgba,group=2,**kw)
def limb(b,name,a,z,r,f,col):
    person(b,name,'capsule',f,col,fromto=(*a,*z),size=r)
person(trunk,'pelvis','ellipsoid',.12,pants,size=(.036,.058,.030))
person(trunk,'torso','ellipsoid',.35,fabric,size=(.039,.063,.09),pos=(0,0,.087))
limb(trunk,'neck',(0,0,.162),(0,0,.188),.014,.015,skin)
person(trunk,'head','ellipsoid',.075,skin,size=(.032,.029,.039),pos=(.006,0,.217))
add(trunk,'geom',name='hair',type='ellipsoid',size=(.030,.028,.019),pos=(.003,0,.238),rgba=(.14,.11,.08,1),mass=0)
for sign,s in [(1,'L'),(-1,'R')]:
    y=.037*sign
    limb(bodies['seat'],'thigh'+s,hip-origin['seat']+[0,y,0],(0,y,0),.021,.105,pants)
    limb(bodies['frame'],'shin'+s,(.280,y,.260),(.308,y,.142),.015,.05,pants)
    limb(bodies['frame'],'shoe'+s,(.309,y,.140),(.337,y,.139),.011,.018,shoe)
    limb(trunk,'upper_arm'+s,(0,.058*sign,.150),(.015,.068*sign,.074),.013,.024,fabric)
    limb(trunk,'forearm'+s,(.015,.068*sign,.074),(.09,.068*sign,.068),.011,.023,skin)
equal=add(root,'equality')
for dep,ratio in [('lower_hinge',1),('back_hinge',-1),('hip_compensation',-1)]:
    add(equal,'joint',joint1=dep,joint2='lift',polycoef=(0,ratio,0,0,0),solref=(.004,1),solimp=(.999,.9999,.001))
add(equal,'connect',name='link_pin',site1='rocker_pin',site2='link_tip',solref=(.004,1),solimp=(.999,.9999,.0001))
motors=add(root,'actuator')
add(motors,'position',name='screw_drive',joint='extension',kp=800000,kv=6000,forcelimited='true',forcerange=(-335.1032164,335.1032164),ctrllimited='true',ctrlrange=(0,.128))
for side in ['L','R']:
    add(motors,'motor',name='JGY_'+side,joint='wheel'+side,ctrllimited='true',ctrlrange=(-.588399,.588399))
ET.indent(root);(HERE/'R4_Gruzdy.xml').write_text(ET.tostring(root,encoding='unicode')+'\n')
