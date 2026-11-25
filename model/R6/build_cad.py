"""Build the R6 bench prototype in mm from R2 and the original Gruzdy STEP assembly.
Run with the Python interpreter bundled with FreeCAD. All bodies are in seated coordinates.
"""
import argparse
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/Applications/FreeCAD.app/Contents/Resources/lib")
import FreeCAD as App
import Part
import MeshPart

V = App.Vector
Y = V(0, 1, 0)
A, B, C = V(280, 0, 260), V(72, 0, 200), V(280, 0, 160)
E, F = V(45, -32, 40), V(182, -32, 232)
L0 = (F - E).Length
U = (F - E) / L0
GAMMA = math.degrees(math.atan2(U.z, U.x))
# Rear pin is on the output rod line, 15 mm aft of STEP axial zero.
# STEP +Y is axial, +Z points to the motor side, +X is the remaining transverse axis.
def installed(x, y, z):
    lateral = z + 64.89
    normal = x + 1.10
    along = y + 15
    return E + U * along + V(-U.z, 0, U.x) * normal + Y * lateral

M = App.Matrix()
o = installed(0, 0, 0)
cols = [installed(*k) - o for k in [(1,0,0), (0,1,0), (0,0,1)]]
M.A = [val for row in range(3) for val in [*(tuple(col)[row] for col in cols), tuple(o)[row]]] + [0,0,0,1]
assert M.determinant() > .999

def turn(p, origin, angle):
    return origin + App.Rotation(Y, angle).multVec(p - origin)

def pose(a):
    tip = turn(F, A, a)
    vec = tip - E
    return tip, vec.Length - L0, GAMMA - math.degrees(math.atan2(vec.z, vec.x))

def tube(a, b, r):
    d = b - a
    return Part.makeCylinder(r, d.Length, a, d)

def box(x, y, z, dx, dy, dz):
    return Part.makeBox(dx, dy, dz, V(x, y, z))

def eye(p, radius, bore, width):
    a = p - Y * (width/2)
    return tube(a, a + Y * width, radius).cut(tube(a-Y, a+Y*(width+1), bore))

def bar(a, b, radius, thickness, bore=0):
    # A flat-ended link with a semicircle at each end, extruded across the pin axis.
    d=b-a
    n=V(-d.z,0,d.x)
    n.normalize()
    n*=radius
    lo=-Y*(thickness/2)
    pts=[a+n+lo,b+n+lo,b-n+lo,a-n+lo]
    core=Part.Face(Part.makePolygon(pts+[pts[0]])).extrude(Y*thickness)
    result=core.fuse(tube(a+lo,a-lo,radius)).fuse(tube(b+lo,b-lo,radius))
    if bore:
        for p in (a,b):
            result=result.cut(tube(p+lo-Y,p-lo+Y,bore))
    return result.removeSplitter()

def merge(*items):
    return items[0].multiFuse(list(items[1:])).removeSplitter() if len(items)>1 else items[0]

def moved(s, delta):
    result=s.copy()
    result.translate(delta)
    return result

def rot(s, origin, deg):
    result=s.copy()
    result.rotate(origin,Y,deg)
    return result

def drill_y(s, x, z, radius):
    return s.cut(tube(V(x,-170,z),V(x,170,z),radius))

def world_raw(shape):
    result=shape.copy()
    result.transformShape(M)
    return result

colors={"frame":[53,111,127],"pad":[53,64,75],"steel":[158,171,176],"tire":[43,50,54],
        "mount":[45,99,104],"battery":[48,57,65],"case":[77,124,150],"gear":[212,151,65],
        "rod":[206,210,212],"eye":[198,112,60],"carriage":[170,91,64],"motor":[104,112,117]}
parts={}
origins={"frame":V(), "seat":A, "back":B, "link":C, "housing":E, "carriage":E}
axes={}
def put(name,shape,group,material,label=None,mass=None,printable=False):
    if shape.isNull() or not shape.isValid() or shape.Volume<=0:
        raise ValueError("Invalid solid: "+name)
    density={"steel":7850,"rod":7850,"tire":1050,"gear":900,"battery":1000,"pad":230}.get(material,650)
    parts[name]=(shape,group,material,label or name,mass if mass is not None else shape.Volume*density*1e-9,printable)

def build(source, actuator):
    base=App.openDocument(str(source))
    for name,group,mat in [('seat','seat','frame'),('back','back','frame'),('link_left','link','frame'),
                             ('link_right','link','frame'),('SeatPad','seat','pad'),('BackPad','back','pad'),
                             ('ArmPadLeft','back','pad'),('ArmPadRight','back','pad'),
                             ('ControllerCase','back','battery'),('ControllerBoot','back','tire'),
                             ('ControllerStick','back','motor'),('ControllerButtons','back','gear')]:
        obj=base.getObject(name)
        shape=obj.Shape.copy()
        if name=='seat':
            # The deck is supported by two side cheeks and ribs beneath the load patch.
            deck=box(78,-94,243.5,180,188,5)
            ribs=[box(98,y,234.5,147,4,10) for y in (-72,68)]
            ribs.extend(box(x,-92,236,4,184,8.5) for x in (167,194))
            ribs.extend(box(94,y,225,155,4,20) for y in (-69,53))
            ribs.extend(box(x,-92,229,5,184,16) for x in (102,200))
            ribs.append(box(167,8,225,5,84,20))
            guide=tube(V(72,-92,200),V(72,92,200),6)
            cheeks=[guide]
            for y in (-94,94):
                aft=V(72,y,200)
                fore=V(280,y,260)
                heel=V(95,y,243)
                toe=V(248,y,243)
                cheeks.append(merge(bar(aft,heel,7,6),bar(heel,toe,7,6),
                                    bar(toe,fore,7,6),eye(aft,9,2.65,6),eye(fore,10,2.65,6)))
            shape=merge(deck,*ribs,*cheeks)
            for x,z in ((72,200),(280,260)):
                shape=drill_y(shape,x,z,2.65)
            # Keep flange contact at Z244 and the four R4 attachment holes.
            shape=shape.cut(box(168.8,-55.2,236,26.4,46.4,8))
            for x in (173,191):
                for y in (-49,-15):
                    shape=shape.cut(tube(V(x,y,239),V(x,y,250),1.65))
            shape=shape.removeSplitter()
        elif name=='SeatPad':
            shape=box(76,-90,248.5,181,180,8)
            shape=shape.makeFillet(3,shape.Edges)
        put(name,shape,group,mat,obj.Label,printable=mat=='frame')
    for name in ['HW_frame_upper','HW_frame_lower','HW_back_lower','SP_frame_upper','SP_frame_lower','SP_back_lower','SP_back_upper']:
        for side in ('L','R'):
            obj=base.getObject(name+'_'+side)
            group='frame' if 'frame' in name else 'back'
            put(obj.Name,obj.Shape.copy(),group,'steel',obj.Label)
    for side in ('L','R'):
        obj=base.getObject('HW_back_join_'+side)
        put(obj.Name,obj.Shape.copy(),'back','steel','Крепёж составной спинки '+side)
    obj=base.getObject('HW_back_upper_through')
    put(obj.Name,obj.Shape.copy(),'back','steel',obj.Label)
    for side,sg in [('L',1),('R',-1)]:
        fork=V(251,sg*100,149)
        rear=V(0,sg*105,64)
        front=V(239,sg*100,64)
        origins.update({'drive'+side:rear,'fork'+side:fork,'front'+side:front})
        for prefix,wheel,group,delta in [
            ('0',rear,'drive'+side,V()),('280',front,'front'+side,V(-41,-5*sg,0))]:
            suffix=prefix+('1' if sg==1 else '_1')
            sourcewheel=base.getObject('_06_wheel_rim'+suffix).Shape.copy()
            rim=moved(sourcewheel,delta)
            if prefix=='0':
                # Integrate the hub into the rim and broach the complete length in one cut.
                center=tube(V(0,sg*88,64),V(0,sg*122,64),9)
                blank=merge(rim,center)
                d_bore=tube(V(0,sg*87,64),V(0,sg*123,64),3.12).cut(box(2.62,-130,54,12,260,20))
                # Slide an M3 nut into the inboard face. A radial M3x6 screw bears on the flat.
                nut_y=93*sg
                radial=lambda x: V(x,nut_y,64)
                tool=tube(radial(1),radial(22),1.65)
                slot=box(5.9,min(86*sg,93*sg),61.15,2.65,7,5.7)
                def hex_prism(x,thickness,across_flats):
                    r=across_flats/math.sqrt(3)
                    vertices=[V(x,nut_y+r*math.cos(k*math.pi/3),64+r*math.sin(k*math.pi/3)) for k in range(6)]
                    return Part.Face(Part.makePolygon(vertices+[vertices[0]])).extrude(V(thickness,0,0))
                rim=blank.cut(d_bore).cut(tool).cut(slot).cut(hex_prism(5.9,2.65,5.7)).removeSplitter()
                nut=hex_prism(6,2.4,5.5).cut(tube(radial(5),radial(10),1.5))
                screw=tube(radial(2.5),radial(8.5),1.48)
                put('HubNut'+side,nut,group,'steel','Гайка M3 '+side)
                put('HubScrew'+side,screw,group,'steel','Стопорный винт M3×6 '+side)
            else:
                rim=drill_y(rim,239,64,8.1)
            put('Rim'+prefix+side,rim,group,'frame','Диск '+group,mass=.094+(rim.Volume-sourcewheel.Volume)*650e-9,printable=True)
            rubber=moved(base.getObject('_07_wheel_tread_TPU'+suffix).Shape.copy(),delta)
            if prefix=='280':
                # Front swivel needs a crowned band. Match the R2 rim Ø120 and overall Ø128.
                edge=64*math.sqrt(1-(17/56)**2)
                a0,a1,at,a2,a3=[V(0,y,z) for y,z in [(-17,60),(-17,edge),(0,64),(17,edge),(17,60)]]
                wire=Part.Wire([Part.makeLine(a0,a1),Part.Arc(a1,at,a2).toShape(),
                                Part.makeLine(a2,a3),Part.makeLine(a3,a0)])
                rubber=moved(Part.Face(wire).revolve(V(),Y,360),wheel)
            put('Tire'+prefix+side,rubber,group,'tire','Шина '+group,printable=True)
        # Direct rear drive. Shaft axis at Z64, gearbox and can retain their native dimensions.
        yo=sg*76
        put('Gearbox'+side,box(-8,min(yo-11,yo+11),48,46,22,32),'frame','motor','JGY-370 · редуктор '+side,.077)
        put('Can'+side,tube(V(38,yo,64),V(69,yo,64),12.2),'frame','motor','JGY-370 · мотор '+side,.043)
        ax=tube(V(0,sg*87,64),V(0,sg*105,64),3)
        flat=box(2.5,-130,54,12,260,20)
        dshaft=ax.cut(flat)
        put('Dshaft'+side,dshaft,'drive'+side,'rod','D-вал 6 мм '+side,.003)
        end=V(239,sg*100,64)
        cheek_y=sg*77
        forkarm=bar(V(251,cheek_y,131),V(239,cheek_y,64),6,6)
        forkarm=drill_y(forkarm,239,64,3.1)
        crown=box(245,min(cheek_y,sg*100)-3,130,12,abs(sg*100-cheek_y)+6,7)
        king=tube(V(251,sg*100,132),V(251,sg*100,153),3.0)
        put('CasterFork'+side,merge(forkarm,crown,king),'fork'+side,'mount','Вилка с вылетом 12 мм '+side,printable=True)
        put('CasterAxle'+side,tube(V(239,sg*74,64),V(239,sg*119,64),3),'fork'+side,'rod','Ось ролика '+side)
        put('CasterHub'+side,tube(V(239,sg*81,64),V(239,sg*117,64),8).cut(
            tube(V(239,sg*79,64),V(239,sg*119,64),3.15)),'front'+side,'eye','Втулка ролика '+side,printable=True)
        # Separate rails and rises allow assembly around battery and drive.
        ry=sg*70
        rail=box(-58,ry-5,34,208,10,12)

        tower=merge(bar(V(131,ry,43),V(191,ry,165),7,10),
                    bar(V(191,ry,165),V(280,ry,160),7,10),
                    bar(V(280,ry,160),V(280,ry,260),7,10))
        for z in (160,260):tower=drill_y(tower,280,z,4.3)
        put('Tower'+side,tower,'frame','frame','Опора четырёхзвенника '+side,printable=True)
        mount=box(-8,sg*63-2,39,44,4,47)
        for xx in (-4,29):
            for zz in (55,73):
                mount=drill_y(mount,xx,zz,1.65)
        # A continuous fillet block stays below the gearbox and stiffens the plate root.
        gusset=box(-8,min(sg*62,sg*70),36,44,8,7)
        side_rail=merge(rail,mount,gusset)
        put('Rail'+side,side_rail,'frame','mount','Балка с моторной пластиной '+side,printable=True)
    # Rear crossmember fixes length to 411.8 mm with the foot platform.
    put('RearBeam',box(-66,-73,34,10,146,12),'frame','frame','Задняя поперечина',printable=True)
    cross=box(244,-106,149,14,212,10)
    for sign in (-1,1):
        cross=cross.cut(tube(V(251,sign*100,146),V(251,sign*100,162),3.15))

    # Front plate is above the swivel envelope. Struts stay inside the gap between the forks.
    footplate=box(299,-71,119,46.815,142,8)
    for sign in (-1,1):
        footplate=merge(footplate,bar(V(251,sign*36,153),V(314,sign*36,123),4,6))
    put('Footplate',merge(footplate,cross),'frame','mount','Подножка с поперечиной вилок',printable=True)
    knee=box(273,-62,208,12,124,22)
    for sign in (-1,1):
        knee=merge(knee,bar(V(274,sign*60,166),V(279,sign*60,217),4,6),
                   box(272,min(sign*60,sign*70)-2,163,7,14,6))
    put('Knee',knee,'frame','pad','Коленный упор',mass=.095,printable=True)
    put('Battery',box(-53,-75.5,105,65,151,97.5),'frame','battery','Yuasa NP7-12 · 12 В · 7 А·ч',2.2)
    tray=box(-57,-80,100,73,160,5)
    for sign in (-1,1):
        tray=merge(tray,bar(V(-50,sign*70,44),V(-50,sign*70,102),4,6),
                   bar(V(10,sign*70,44),V(10,sign*70,102),4,6))
    put('BatteryTray',tray,'frame','mount','Лоток аккумулятора',printable=True)
    # Pin E and external ears leave clearance for the pivoting rear cover.
    saddle=box(36,-75,19,17,150,6)
    for sign in (-1,1):
        saddle=merge(saddle,box(36,sign*70-4,24,17,8,12))
    for offset in (-11,11):
        ear=bar(E+Y*offset+V(0,0,-16),E+Y*offset,6.5,4)
        saddle=merge(saddle,ear)
    put('LowerMount',drill_y(saddle,E.x,E.z,3.15),'frame','mount','Новая нижняя вилка E',printable=True)
    # Upper pin F lies below the R2 pad. Plate attaches to the underside of the seat.
    upper=box(169,-55,240.5,26,46,3.5)
    for x in (173,191):
        for y in (-49,-15):upper=upper.cut(tube(V(x,y,239),V(x,y,248),1.65))
    for off in (-9,9):
        ear=bar(F+Y*off,F+Y*off+V(0,0,9),6,4)
        upper=merge(upper,ear)
    put('UpperMount',drill_y(upper,F.x,F.z,3.15),'seat','mount','Новая верхняя вилка F',printable=True)
    for lab,p,body in [('BasePin',E,'frame'),('TipPin',F,'seat')]:
        put(lab,tube(p-Y*15,p+Y*15,3),body,'rod','Палец Ø6 '+lab)
    solids=Part.read(str(actuator)).Solids
    if len(solids)!=8:raise ValueError('Expected the eight placed solids of Gruzdy')
    # Front of seated carriage at STEP axial coordinate 47.4, rear at 32.0.
    cart_shift=32.0-solids[4].BoundBox.YMin
    groups=['housing','housing','driven','idler','carriage','pinion','shaft','housing']
    labels=['Задняя крышка','Корпус','Выходная шестерня 48','Промежуточная шестерня 48',
            'Каретка','Шестерня двигателя 12','Винт Ø10, гладкая CAD-поверхность','Передняя крышка']
    for name,xyz in {'pinion':(-1.08,12,-.08),'idler':(-1.08,12,-16.93),'driven':(-1.08,12,-43.91),
                     'shaft':(-1.339,12,-43.76)}.items():
        origins[name]=installed(*xyz)
    for i,(s,group) in enumerate(zip(solids,groups)):
        shape=s.copy()
        if i==4:shape.translate(V(.45,cart_shift,-.39))
        mat=['case','case','gear','gear','carriage','gear','steel','case'][i]
        put('Gruzdy'+str(i),world_raw(shape),group,mat,labels[i],printable=i in (0,1,2,3,4,5,7))
    m1,m2=installed(-1.09,26,0),installed(-1.09,55.5,0)
    put('Stepper',tube(m1,m2,24.45),'housing','motor','M49SP-2K · корпус',.22)
    # Adapter on the existing four Ø5 rear-cover holes. Clear the rear pin and screw axes.
    flange=box(-37.5,-6,-75,73,4,66)
    for xx in (-33.09,30.91):
        for zz in (-70.26,-14.42):
            flange=flange.cut(tube(V(xx,-8,zz),V(xx,0,zz),2.6))
    # Two struts sit behind the cover; eye is on the rod axis.
    adaptor=merge(world_raw(flange),tube(E+U*2,installed(-1.1,-5,-64.89),5.5),eye(E,7,3.1,13))
    put('RearAdapter',drill_y(adaptor,E.x,E.z,3.1),'housing','eye','Адаптер задней крышки',printable=True)
    # Rigid output rod passes through the end cover's Ø10 bore. A removable eye is fitted at its tip.
    def captive_hex(x,z,bore):
        radius=16/math.sqrt(3)
        pts=[V(x+radius*math.cos(math.radians(30+60*k)),35.5,z+radius*math.sin(math.radians(30+60*k))) for k in range(6)]
        blank=Part.Face(Part.makePolygon(pts+pts[:1])).extrude(V(0,8,0))
        return world_raw(blank.cut(tube(V(x,34,z),V(x,45,z),bore)))
    put('LeadscrewNut',captive_hex(-1.339,-43.76,5.05),'carriage','steel','Гайка M10×1,5 · SW16 · условная резьба')
    put('RodBush',captive_hex(-1.10,-64.89,4.54),'carriage','steel','Втулка штока · SW16')
    rod=tube(installed(-1.10,35.5,-64.89),F-U*5,4.5)
    put('OutputRod',rod,'carriage','rod','Выходной шток Ø9',mass=rod.Volume*7.85e-6)
    put('RodEye',merge(eye(F,6.8,3.1,12),tube(F-U*10,F-U*3,5)).cut(tube(F-Y*10,F+Y*10,3.1)),
        'carriage','eye','Проушина штока',printable=True)
    revise_chassis()
    App.closeDocument(base.Name)

def revise_chassis():
    """Create mating faces before drilling both members in assembly coordinates."""
    def shape(name):
        return parts[name][0]

    def update(name, solid):
        old, group, material, label, mass, printable = parts[name]
        put(name, solid.removeSplitter(), group, material, label,
            mass * solid.Volume / old.Volume, printable)

    def hexagon(center, axis, across, length):
        rotation=App.Rotation(V(0,0,1),axis)
        corners=[center+rotation.multVec(V(across/math.sqrt(3)*math.cos(k*math.pi/3),
                                           across/math.sqrt(3)*math.sin(k*math.pi/3),0)) for k in range(6)]
        return Part.Face(Part.makePolygon(corners+corners[:1])).extrude(axis*length)

    def through_joint(label, member, support, top, axis, grip, diameter, length, group='frame'):
        # Axis runs from the outside of the flush head towards the nut.
        r=diameter/2
        head_height=r
        h=diameter*.8
        clearance=tube(top-axis,top+axis*(grip+1),r+.2)
        countersink=Part.makeCone(diameter+.18,r+.18,head_height,top,axis)
        update(member,shape(member).cut(clearance).cut(countersink))
        update(support,shape(support).cut(clearance))
        fastener=merge(Part.makeCone(diameter,r,head_height,top,axis),
                       tube(top+axis*head_height,top+axis*length,r))
        nut_pos=top+axis*grip
        nut=hexagon(nut_pos,axis,7 if diameter==4 else 5.5,h)
        nut=nut.cut(tube(nut_pos-axis,nut_pos+axis*(h+1),r+.02))
        put(label+'Bolt',fastener,group,'steel',f'M{diameter}×{length} · '+label)
        put(label+'Nut',nut,group,'steel',f'Гайка M{diameter} · '+label)

    down=V(0,0,-1)
    # Deep crossbeam below the pin, with thick webs rising inboard of the rails.
    saddle=box(32,-64.8,11,31,129.6,15)
    for y in (-42.5,-21.5):
        saddle=merge(saddle,bar(V(45,y,23),V(45,y,40),7.5,6))
    for sign in (-1,1):
        def edge(a,b): return min(a*sign,b*sign)
        # Inboard web, top bearing face and longitudinal shear key.
        saddle=merge(saddle,box(38,edge(60.8,64.8),18,29,4,33),
                     box(38,edge(60.8,75.5),46,39,14.7,5))
        for x in (50,69):
            saddle=merge(saddle,box(x,edge(66.3,73.7),41.8,5,7.4,4.3))
    roots=[e for e in saddle.Edges if abs(e.BoundBox.ZMin-46)<1e-5 and abs(e.BoundBox.ZMax-46)<1e-5 and
           49.99<=e.BoundBox.XMin<=74.01 and e.BoundBox.XMax<=74.01 and 66.29<=abs(e.CenterOfMass.y)<=73.71]
    saddle=saddle.makeFillet(.7,roots)
    # Clear the integrated motor plates during vertical installation between the rails.
    for y in (-80,59):
        saddle=saddle.cut(box(27,y,7,10.5,21,25))
    update('LowerMount',drill_y(saddle,E.x,E.z,3.15))
    # Battery bridge stands clear of both motors, and ties the rear of the rails.
    tray=box(-63,-80,100,83,160,5)
    for x in (-59,15):
        tray=tray.cut(box(x,-11.5,99,3.2,23,7))
    tray=merge(tray,box(-66,-75,46,12,150,5))
    for sign in (-1,1):
        y=sign*70
        tray=merge(tray,box(-56,y-5,46,45,10,5),
                   bar(V(-50,y,50),V(-50,y,101),4.5,8),
                   bar(V(-18,y,50),V(-18,y,101),4.5,8))
    del parts['RearBeam']
    tray=tray.cut(box(-70,-85,105,95,170,8))
    tray=tray.cut(box(-60,-78,32,55,156,14))
    update('BatteryTray',tray)
    knee=box(273,-62,208,12,124,22)
    for sign in (-1,1):
        y=sign*60
        knee=merge(knee,bar(V(279,y,185),V(279,y,215),4,6),
                   box(273,min(sign*58,sign*65),176,12,7,23))
    update('Knee',knee)
    for side,sign in [('L',1),('R',-1)]:
        y=70*sign
        rail='Rail'+side
        tower='Tower'+side
        for x in (50,69):
            groove=merge(box(x-.2,y-3.9,41.6,5.4,7.8,5),box(x-.9,y-4.6,45.25,6.8,9.2,1))
            update(rail,shape(rail).cut(groove))
        for i,x in enumerate((43,63)):
            through_joint(f'Mount{side}{i}', 'LowerMount',rail,V(x,y,51),down,17,4,22)
        for i,x in enumerate((-55,-30)):
            through_joint(f'Tray{side}{i}', 'BatteryTray',rail,V(x,y,51),down,17,3,22)
        post=shape(tower).cut(box(105,y-8,0,65,16,46))
        post=merge(post,box(115,y-5,46,35,10,7),box(77.5,y-5,46,39.5,10,7),
                   bar(V(94,y,53),V(280,y,239),8,10),
                   bar(V(93,y,54),V(250,y,166),7,10))
        post=post.cut(box(75,y-8,32,81,16,14))
        post=drill_y(post,280,260,4.3)
        # The front caster bridge bears against the underside of the post.
        post=post.cut(box(243,y-7,139,17,14,20))
        post=merge(post,box(244,y-5,159,14,10,13))
        update(tower,post)
        for i,x in enumerate((81.5,114)):
            through_joint(f'Brace{side}{i}',rail,tower,V(x,y,34),-down,19,4,25)
        for i,x in enumerate((120,145)):
            through_joint(f'Post{side}{i}',tower,rail,V(x,y,53),down,19,4,25)
        through_joint('Front'+side,tower,'Footplate',V(251,y,172),down,23,4,28)
        for i,z in enumerate((182,194)):
            through_joint(f'Knee{side}{i}','Knee',tower,V(279,58*sign,z),Y*sign,17,3,22)
        # Rear tread has a molded seat matching the rim. Relieve the inboard tread lip.
        tire='Tire0'+side
        tire_shape=shape(tire).cut(shape('Rim0'+side))
        inner=box(-66,min(sign*87,sign*88.8),-2,132,1.8,132)
        inner=inner.cut(tube(V(0,87*sign,64),V(0,90*sign,64),18))
        update('Rim0'+side,shape('Rim0'+side).cut(inner))
        tire_shape=tire_shape.cut(box(-66,min(sign*87,sign*88.8),-2,132,1.8,132))
        update(tire,tire_shape)

        # Model the motor's four tapped fixing holes and screws through its plate.
        for i,x in enumerate((-4,29)):
            for j,z in enumerate((55,73)):
                end=V(x,61*sign,z)
                into=Y*sign
                update('Gearbox'+side,shape('Gearbox'+side).cut(tube(end+into*3,end+into*9.5,1.5)))
                screw=merge(tube(end,end+into*9,1.45),tube(end-into*2,end,2.7))
                put(f'Motor{side}{i}{j}',screw,'frame','steel','Винт моторной пластины M3')
        # A retained steel kingpin replaces the short printed peg.
        fork='CasterFork'+side
        pinbase=V(251,100*sign,130)
        axis=V(0,0,1)
        fork_shape=shape(fork).cut(box(246,100*sign-5,137,10,10,23))
        fork_shape=fork_shape.cut(tube(pinbase-axis,pinbase+axis*9,3.2))
        fork_shape=fork_shape.cut(Part.makeCone(5.7,3.2,2.5,pinbase,axis))
        update(fork,fork_shape)
        def ring(center,axis,length,outer,inner):
            return tube(center,center+axis*length,outer).cut(tube(center-axis,center+axis*(length+1),inner))
        king=merge(Part.makeCone(5.5,3,2.5,pinbase,axis),tube(pinbase+axis*2.5,pinbase+axis*36,3))
        put('Kingpin'+side,king,'fork'+side,'steel','Стальная ось поворота Ø6')
        put('KingSpacer'+side,ring(pinbase+axis*7,axis,11.8,4.8,3.1),'fork'+side,'steel','Дистанционная втулка')
        put('KingWasher'+side,ring(pinbase+axis*29.2,axis,.8,5.5,3.1),'fork'+side,'steel','Шайба оси поворота')
        nut=hexagon(pinbase+axis*30,axis,10,5.5).cut(tube(pinbase+axis*29,pinbase+axis*36,3.02))
        put('KingNut'+side,nut,'fork'+side,'steel','Самоконтрящаяся гайка M6')
        axle=V(239,74*sign,64)
        out=Y*sign
        axle_shape=merge(tube(axle,axle+out*50,3),tube(axle-out*2,axle,4.8))
        update('CasterAxle'+side,axle_shape)
        nut=hexagon(axle+out*43.3,out,10,5.5).cut(tube(axle+out*42,out*50+axle,3.02))
        put('AxleNut'+side,nut,'fork'+side,'steel','Самоконтрящаяся гайка оси колеса M6')
    # Closed seats for the upper clevis and flush hardware below the cushion.
    update('seat',shape('seat').cut(shape('UpperMount')))
    for i,x in enumerate((173,191)):
        for j,y in enumerate((-49,-15)):
            through_joint(f'TipPlate{i}{j}','seat','UpperMount',V(x,y,248.5),down,8,3,12,'seat')

    # Socket the output rod and retain it with a transverse pin.
    socket=merge(eye(F,6.8,3.1,12),tube(F-U*15,F-U*4,6.3))
    socket=socket.cut(tube(F-U*17,F-U*7,4.65))
    pin=F-U*11
    hole=tube(pin-Y*12,pin+Y*12,1.35)
    update('RodEye',drill_y(socket,F.x,F.z,3.1).cut(hole))
    update('OutputRod',tube(installed(-1.10,35.5,-64.89),F-U*7.2,4.5).cut(hole))
    put('RodCrossPin',tube(pin-Y*6,pin+Y*6,1.25),'carriage','steel','Поперечный штифт Ø2,5×12')
    # Phase the straight teeth around their own bore. The native placed STEP has tip-on-tip overlap.
    idler=shape('Gruzdy3').copy()
    idler.rotate(origins['idler'],U,3.75)
    update('Gruzdy3',idler)
    pinion=shape('Gruzdy5').copy()
    pinion.rotate(origins['pinion'],U,-2)
    update('Gruzdy5',pinion)
    for i,x in enumerate((-33.09,30.91)):
        for j,z in enumerate((-70.26,-14.42)):
            start=installed(x,-6,z)
            axis=U
            for name in ('RearAdapter','Gruzdy0'):
                update(name,shape(name).cut(tube(start-axis,start+axis*10,2.65)))
            screw=merge(tube(start-axis*2.7,start,4.3),tube(start,start+axis*14,2.48))
            put(f'Adapter{i}{j}Bolt',screw,'housing','steel','Винт M5×14')
            nut=hexagon(start+axis*9,axis,8,4).cut(tube(start+axis*8,start+axis*14,2.5))
            put(f'Adapter{i}{j}Nut',nut,'housing','steel','Гайка M5')
    # The 48T idler runs on a shouldered shaft fixed through the rear plate.
    center=installed(-1.0795,-6,-16.93197)
    for name in ('RearAdapter','Gruzdy0'):
        update(name,shape(name).cut(tube(center-U,center+U*9,1.6)))
    axle=merge(tube(center-U*3,center+U*23.15,1.49),tube(center+U*23.15,center+U*24.75,2.7))
    put('IdlerAxle',axle,'housing','steel','Ось промежуточной шестерни Ø3')
    put('IdlerNut',hexagon(center,-U,5.5,2.4).cut(tube(center-U*3,center+U,1.51)),'housing','steel','Гайка оси M3')
    put('MotorSpindle',tube(installed(-1.0795,7,-.078),installed(-1.0795,26,-.078),1.49),'pinion','steel','Вал шагового двигателя Ø3')
    # Provide running clearance through the housing around the actual screw axis.
    bore=tube(installed(-1.0795,18,-43.91055),installed(-1.0795,24,-43.91055),5.25)
    update('Gruzdy1',shape('Gruzdy1').cut(bore))
    for name,center,width,group in [('BasePin',E,27,'frame'),('TipPin',F,22,'seat')]:
        start=center-Y*(width/2+.3)
        clip=start+Y*(width+2.8)
        pin=merge(tube(start,start+Y*(width+5),3),tube(start-Y*2,start,4.8))
        update(name,pin.cut(tube(clip-V(0,0,5),clip+V(0,0,5),.85)))
        put(name+'Clip',tube(clip-V(0,0,4.3),clip+V(0,0,4.3),.75),group,'steel','Поперечный шплинт Ø1,5')
    # Keep the screw coaxial with the output gear. Machine its front end to a Ø6 journal.
    sx,sz=-1.0795,-43.91055
    shift=installed(sx,0,sz)-installed(-1.339,0,-43.76)
    origins['shaft']=installed(sx,12,sz)
    def axial(x,z,lo,hi,r):
        return tube(installed(x,lo,z),installed(x,hi,z),r)
    shaft=moved(shape('Gruzdy6'),shift).cut(axial(sx,sz,179.3,196,13))
    update('Gruzdy6',merge(shaft,axial(sx,sz,179.2,211,3)))
    update('LeadscrewNut',moved(shape('LeadscrewNut'),shift))
    boss=axial(sx,sz,180.8,203.5,9.7)
    cap=merge(shape('Gruzdy7'),boss).cut(axial(sx,sz,178,204,5.1))
    cap=cap.cut(axial(sx,sz,178,182.3,8.15))
    update('Gruzdy7',cap)
    update('Gruzdy1',shape('Gruzdy1').cut(axial(sx,sz,180.7,184,9.85)))
    bush=merge(axial(sx,sz,180.8,182.3,8),axial(sx,sz,182.3,203.5,5))
    update_bore=axial(sx,sz,177,213,3.12)
    put('ThrustBush',bush.cut(update_bore),'housing','eye','Фланцевая металлическая втулка Ø10/6,24',mass=bush.cut(update_bore).Volume*8.5e-6)
    for name,lo,hi in [('Inner',179.3,180.8),('Outer',203.5,205)]:
        put('ThrustWasher'+name,axial(sx,sz,lo,hi,7).cut(update_bore),'shaft','steel','Упорная шайба Ø14/6,24')
    nut=hexagon(installed(sx,205.1,sz),U,10,5.5).cut(update_bore)
    put('ThrustNut',nut,'shaft','steel','Стопорная гайка M6 на цапфе')
    # Four M3 tie rods clamp the thrust cap, main case and rear clevis flange together.
    for i,(x,z) in enumerate(((-25.59,-29.99),(23.41,-29.99),(-22.66,-77.46),(20.47,-77.46))):
        hole=axial(x,z,-12,211,1.7)
        for name,lo,hi in [('RearAdapter',-6,-2),('Gruzdy7',199.5,203.5)]:
            update(name,merge(shape(name),axial(x,z,lo,hi,4.8)).cut(hole))
        for name in ('Gruzdy1','Gruzdy0'):
            update(name,shape(name).cut(hole))
        put(f'CaseTie{i}',axial(x,z,-10.8,208.3,1.48),'housing','steel','Стяжная шпилька M3, отрезок 219,1 мм')
        for end,y,direction in [('Back',-6,-U),('Front',203.5,U)]:
            pt=installed(x,y,z)
            washer=ring(pt,direction,.8,3.5,1.6)
            nut=hexagon(pt+direction*.8,direction,5.5,2.4).cut(hole)
            put(f'CaseTie{i}{end}Washer',washer,'housing','steel','Шайба M3')
            put(f'CaseTie{i}{end}Nut',nut,'housing','steel','Гайка стяжки M3')
    # Split off the front face so both captive inserts can be installed and removed.
    cart=shape('Gruzdy4')
    for x,z,r in ((sx,sz,5.4),(-1.1,-64.89,4.8)):
        for lo,hi in ((32,35.37),(43.63,47.4)):
            cart=merge(cart,axial(x,z,lo,hi,10.1).cut(axial(x,z,lo-.5,hi+.5,r)))
    cart=cart.cut(shape('LeadscrewNut')).cut(shape('RodBush'))
    slab=world_raw(box(-16,43.9,-80,29,6,54))
    lid=cart.common(slab)
    update('Gruzdy4',cart.cut(slab))
    put('CarriageLid',lid,'carriage','carriage','Съёмная крышка каретки',printable=True)
    for i,x in enumerate((-10,7.6)):
        for j,z in enumerate((-74,-35)):
            tip=installed(x,47.4,z)
            drill=axial(x,z,27,48,1.38)
            update('Gruzdy4',shape('Gruzdy4').cut(drill))
            update('CarriageLid',shape('CarriageLid').cut(drill).cut(Part.makeCone(2.55,1.38,1.17,tip,-U)))
            bolt=merge(Part.makeCone(2.5,1.25,1.25,tip,-U),tube(tip-U*1.25,tip-U*19,1.24))
            nut=hexagon(installed(x,32,z),-U,5,2).cut(drill)
            put(f'Cart{i}{j}Bolt',bolt,'carriage','steel','Винт M2,5, длина после подрезки 19 мм')
            put(f'Cart{i}{j}Nut',nut,'carriage','steel','Гайка M2,5')
    p=installed(-1.1,39.5,-64.89)
    normal=V(-U.z,0,U.x)
    drill=tube(p-normal*10,p+normal*10,1.3)
    for name in ('OutputRod','RodBush'):update(name,shape(name).cut(drill))
    put('RodTailPin',tube(p-normal*7.9,p+normal*7.9,1.25),'carriage','steel','Штифт Ø2,5×15,8 в хвостовике штока')
    # Shape the cushions around the frame and the electronics while retaining their outer profile.
    for name,cutters in [('ControllerCase',['ControllerBoot','ControllerButtons']),
                         ('ControllerBoot',['ControllerStick']),
                         ('ArmPadRight',['back','ControllerCase','BackPad']),
                         ('ArmPadLeft',['back','BackPad']),('BackPad',['back'])]:
        solid=shape(name)
        for cutter in cutters:solid=solid.cut(shape(cutter))
        update(name,solid)
    update('SeatPad',shape('SeatPad').cut(box(60,-100,244,38,200,20)))


def placed(shape, group, angle):
    _,travel,pitch=pose(angle)
    if group=='seat':return rot(shape,A,angle)
    if group=='back':return moved(shape,turn(B,A,angle)-B)
    if group=='link':return rot(shape,C,angle)
    if group in ('housing','carriage','driven','idler','pinion','shaft'):
        return rot(moved(shape,U*travel) if group=='carriage' else shape,E,pitch)
    return shape.copy()

def print_orientation(mesh,name):
    if name.startswith(('Rim','Tire','CasterHub','link_','Tower')):
        matrix=App.Matrix()
        matrix.rotateX(math.radians(-90 if name.endswith('L') else 90))
        mesh.transform(matrix)
    bb=mesh.BoundBox
    mesh.translate(-bb.XMin,-bb.YMin,-bb.ZMin)

def hub_coupon(shape,out):
    # A short section of the actual left hub includes the nut slot and set-screw hole.
    sample=shape.common(tube(V(0,88,64),V(0,100,64),12))
    mesh=MeshPart.meshFromShape(Shape=sample,LinearDeflection=.06,AngularDeflection=.15,Relative=False)
    print_orientation(mesh,'Rim0L')
    dest=out/'Проба посадки'
    dest.mkdir(exist_ok=True)
    mesh.write(str(dest/'Ступица_D6_M3.stl'))

def carriage_samples(body,lid,out):
    for shape,name in ((body,'Каретка'),(lid,'Крышка каретки')):
        local=shape.copy()
        local.transformShape(M.inverse())
        local.rotate(V(),V(1,0,0),90)
        mesh=MeshPart.meshFromShape(Shape=local,LinearDeflection=.08,AngularDeflection=.18,Relative=False)
        print_orientation(mesh,name)
        mesh.write(str(out/'Проба посадки'/(name+'.stl')))

def export(out):
    import tempfile
    from copy import deepcopy
    from zipfile import ZipFile,ZIP_DEFLATED
    import xml.etree.ElementTree as ET
    out.mkdir(exist_ok=True,parents=True)
    assets=out/'assets';assets.mkdir(exist_ok=True)
    solids_dir=out/'STL';solids_dir.mkdir(exist_ok=True)
    meta={}
    for name,(s,g,mat,label,mass,printed) in parts.items():
        local=moved(s,-origins[g])
        mesh=MeshPart.meshFromShape(Shape=local,LinearDeflection=.13,AngularDeflection=.22,Relative=False)
        mesh.write(str(assets/(name+'.stl')))
        if printed:
            if name=='back':
                for solid in s.Solids:
                    component='BackUpper' if solid.BoundBox.ZMin>200 else ('BackPostL' if solid.CenterOfMass.y>0 else 'BackPostR')
                    single=MeshPart.meshFromShape(Shape=solid,LinearDeflection=.13,AngularDeflection=.22,Relative=False)
                    bb=single.BoundBox
                    single.translate(-bb.XMin,-bb.YMin,-bb.ZMin)
                    single.write(str(solids_dir/(component+'.stl')))
            else:
                if len(s.Solids)!=1:
                    raise ValueError(f'{name} has {len(s.Solids)} separate solids')
                model=mesh.copy()
                print_orientation(model,name)
                model.write(str(solids_dir/(name+'.stl')))
        meta[name]=dict(group=g,color=colors[mat],mass=mass,label=label)
    hub_coupon(parts['Rim0L'][0],out)
    carriage_samples(parts['Gruzdy4'][0],parts['CarriageLid'][0],out)
    state={'parts':meta,'origins':{k:tuple(v) for k,v in origins.items()},'E':tuple(E),'F':tuple(F),
           'direction':tuple(U),'pin_distance_mm':L0,'stroke_mm':pose(75)[1]}
    (assets/'assembly.json').write_text(json.dumps(state,ensure_ascii=False,indent=1))
    for angle in (0,75):
        doc=App.newDocument('R6');doc.Label=f'R6 · Gruzdy 1:1 · {angle}°'
        groups={}
        for name,(s,g,mat,label,mass,pr) in parts.items():
            cls='LiftDrive' if g in ('housing','pinion','driven','idler','shaft','carriage') else ('Chair' if g in ('seat','back','link') else 'Chassis')
            if cls not in groups:
                gr=doc.addObject('App::DocumentObjectGroup',cls)
                gr.Label={'LiftDrive':'Актуатор Gruzdy','Chair':'Сиденье и четырёхзвенник','Chassis':'Шасси и питание'}[cls]
                groups[cls]=gr
            ob=doc.addObject('Part::Feature',name);ob.Label=label;ob.Shape=placed(s,g,angle);groups[cls].addObject(ob)
        doc.recompute()
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)/'R6.FCStd';doc.saveAs(str(raw));App.closeDocument(doc.Name)
            with ZipFile(raw) as source, ZipFile(args.source) as template, ZipFile(out/f'R6_{angle}deg.FCStd','w',ZIP_DEFLATED) as dest:
                for n in source.namelist():dest.writestr(n,source.read(n))
                gui=ET.fromstring(template.read('GuiDocument.xml'));providers=gui.find('ViewProviderData')
                part=next(x for x in providers if x.find('.//MaterialList') is not None)
                group=next(x for x in providers if x.get('name')=='Base'); providers.clear(); resources={}
                for i,name in enumerate([*groups,*parts]):
                    node=deepcopy(part if name in parts else group);node.set('name',name);node.set('treeRank',str(i+1));providers.append(node)
                    if name not in parts:continue
                    for ref in node.findall('.//*[@file]'):
                        fn=ref.get('file');data=template.read(fn)
                        if ref.tag=='MaterialList':
                            r,gg,b=meta[name]['color'];data=data[:8]+bytes([255,b,gg,r])+data[12:]
                            fn=f'color_{i}';ref.set('file',fn)
                        resources[fn]=data
                    if name in ('Gruzdy0','Gruzdy1','Gruzdy7'):
                        node.find(".//Property[@name='Transparency']")[0].set('value','54')
                providers.set('Count',str(len(providers)))
                cam=gui.find('Camera')
                h=530 if angle==0 else 780
                cam.set('settings',f'OrthographicCamera {{ position 795 -812 {635 if angle==0 else 754} orientation 0.86307 0.29075 0.41301 1.36843 nearDistance 1 farDistance 2600 focalDistance 1100 height {h} }}')
                dest.writestr('GuiDocument.xml',ET.tostring(gui,encoding='utf-8',xml_declaration=True))
                for n,data in resources.items():dest.writestr(n,data)
        print('Saved',angle,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    inputs=HERE/'inputs'
    source=inputs/'Verticalizator_R2.FCStd' if inputs.is_dir() else HERE.parent/'Verticalizator_R2.FCStd'
    actuator=inputs/'Сборка.stp'
    if not actuator.is_file():
        actuator=HERE.parent/'inputs/Сборка.stp'
    if not actuator.is_file():
        actuator=Path.home()/'Gruzdy/models/m3d/Сборка.stp'
    parser.add_argument('--source',type=Path,default=source)
    parser.add_argument('--actuator',type=Path,default=actuator)
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args()
    build(args.source,args.actuator)
    export(args.output)
