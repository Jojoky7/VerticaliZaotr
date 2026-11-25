"""Build the rigid-wheel bench assembly from the seated R6 reference, in mm.

Run with FreeCAD Python and an output directory. The geometry report, world-space
component meshes and group metadata can also be used by the R6 interference tests.
"""
import argparse
import json
import math
from pathlib import Path

from build_cad import App, HERE, MeshPart, Part, V, Y


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    assets = output / 'assets'
    assets.mkdir(exist_ok=True)
    doc = App.openDocument(str(HERE / 'R6_0deg.FCStd'))
    state = json.loads((HERE / 'assets/assembly.json').read_text())
    parts = state['parts']
    wheel_report = {}

    def put(name, shape, group, label, color):
        if len(shape.Solids) != 1 or not shape.isValid():
            raise ValueError(f'Invalid part: {name}')
        obj = doc.getObject(name)
        if obj is None:
            obj = doc.addObject('Part::Feature', name)
            doc.getObject('Chassis').addObject(obj)
        obj.Label = label
        obj.Shape = shape
        parts[name] = dict(group=group, label=label, color=color)

    def remove(name):
        doc.removeObject(name)
        del parts[name]

    steel = [151, 165, 173]
    petg = [48, 113, 127]
    for side, sign in [('L', 1), ('R', -1)]:
        for suffix, key, group, label in (
            ('0', 'Rear', 'drive', 'Цельное заднее колесо PETG'),
            ('280', 'Front', 'front', 'Цельное переднее колесо PETG'),
        ):
            rim = 'Rim' + suffix + side
            tread = 'Tire' + suffix + side
            a, b = (doc.getObject(n).Shape for n in (rim, tread))
            wheel = a.fuse(b).removeSplitter()
            delta = abs(wheel.Volume - a.Volume - b.Volume)
            if delta > 0.1:
                raise ValueError(f'Wheel volume discrepancy: {delta}')
            name = 'Wheel' + key + side
            put(name, wheel, group + side, label + ' ' + side, petg)
            wheel_report[name] = dict(solids=len(wheel.Solids), volume_delta_mm3=delta)
            remove(rim)
            remove(tread)

        def cylinder(x, z, start, end, diameter):
            axis = Y * (sign if end > start else -sign)
            return Part.makeCylinder(diameter / 2, abs(end-start), V(x, sign*start, z), axis)

        def ring(x, z, start, end, outer, inner):
            return cylinder(x,z,start,end,outer).cut(cylinder(x,z,start-1,end+1,inner))

        def nut(x, z, start, end, flats, bore):
            r = flats / math.sqrt(3)
            pts = [V(x+r*math.cos(i*math.pi/3),sign*start,z+r*math.sin(i*math.pi/3)) for i in range(6)]
            blank = Part.Face(Part.makePolygon(pts+pts[:1])).extrude(Y*(sign*(end-start)))
            return blank.cut(cylinder(x,z,start-1,end+1,bore))

        # Clamp the flanged bushing to the fork. The wheel runs on its smooth OD.
        hub = ring(239,64,80,82.8,20,6.3).fuse(ring(239,64,82.8,117.4,16,6.3)).removeSplitter()
        put('CasterHub'+side, hub, 'fork'+side, 'Втулка переднего колеса с буртом '+side, petg)
        axle = cylinder(239,64,72,74,9.6).fuse(cylinder(239,64,74,126,6)).removeSplitter()
        put('CasterAxle'+side, axle, 'fork'+side, 'Ось Ø6 × 52 под головкой '+side, steel)
        put('AxleWasher'+side,ring(239,64,117.4,119,18,6.4),'fork'+side,'Шайба Ø18/6,4 × 1,6 '+side,steel)
        put('AxleNut'+side,nut(239,64,119,124.5,10,6.04),'fork'+side,'Стопорная гайка M6 '+side,steel)

        for z, level, gap, head, length in [(260,'upper',15.8,97.2,40),(160,'lower',41.8,123.2,66)]:
            tag = 'frame_'+level+'_'+side
            put('Sleeve_'+tag,ring(280,z,65,75,8.5,5.2),'frame','Втулка Ø8,5/5,2 × 10 '+tag,steel)
            put('SP_'+tag,ring(280,z,75,75+gap,9.6,5.3),'frame',f'Распорка {gap} мм '+tag,steel)
            pin = cylinder(280,z,head-length,head,5).fuse(cylinder(280,z,head,head+2,8.4)).removeSplitter()
            put('HW_'+tag,pin,'frame',f'Ось Ø5 × {length}, конец M5 '+tag,steel)
            put('Washer_'+tag,ring(280,z,64,65,10,5.3),'frame','Шайба Ø10/5,3 × 1 '+tag,steel)
            put('Nut_'+tag,nut(280,z,58.5,64,8,5.04),'frame','Стопорная гайка M5 '+tag,steel)

    doc.Label = 'R6 • PETG • Сборка'
    doc.recompute()
    doc.saveAs(str(output / 'R6_PETG.FCStd'))
    for name in parts:
        mesh = MeshPart.meshFromShape(Shape=doc.getObject(name).Shape, LinearDeflection=.11,
                                      AngularDeflection=.19, Relative=False)
        mesh.write(str(assets / (name+'.stl')))
    (assets / 'assembly.json').write_text(json.dumps(state, ensure_ascii=False, indent=1))
    (output / 'wheels.json').write_text(json.dumps(wheel_report, indent=2))
    App.closeDocument(doc.Name)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    build(cli.parse_args().output.resolve())
