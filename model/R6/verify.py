"""Run the published geometry, material-model, lift and drive checks; or verify stored reports."""
import argparse
import gzip
import shutil
import json
import subprocess
import sys
import tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
QA=HERE/'verification'
def read(path):return json.loads(path.read_text())
def accept():
    errors=[]
    def check(condition,topic):
        if not condition:errors.append(topic)
    pairs=read(QA/'all_pairs.json')
    check(not pairs['interferences'] and sorted(pairs['angles_deg'])==list(range(0,76,5)), 'All-pairs sweep')
    check(pairs['components']==len(read(HERE/'assets/assembly.json')['parts']), 'CAD component count')
    fit=read(HERE/'fit.json')
    check(not fit['interferences'] and fit['step_deg']==1,'Lift path at 1 degree')
    check(not read(QA/'caster_sweep.json')['envelope_hits'],'Swivel and wheel rolling envelopes')
    check(not read(QA/'slider.json')['interferences'],'Slider, screws and housing')
    check(not read(QA/'joints.json')['issues'],'Mating faces, insertion, fasteners and meshes')
    check(all(row['max_area_mm2']<=.001 for row in read(QA/'gear_cycle.json').values()),'Gear phases')
    cycles=read(QA/'cycles.json')
    check([r['mass_kg'] for r in cycles]==[5,6,8] and all(r['passed'] for r in cycles),'Full lift and return')
    results=read(HERE/'results.json')
    check(all(r['passed'] and r['min_sampled_wheel_load_N']>0 for r in results['on_wheels']),'Lift on four wheels')
    check(results['drive']['forward_2s_mm']>0 and abs(results['drive']['turn_2s_deg'])>10,'Differential steering')
    check(results['half_timestep_8kg']['passed'],'Time-step refinement')
    check(not results['scenarios']['higher_losses']['trials'][-1]['passed'],'Reduced torque stalls at 8 kg')
    for name in ('mount_25','mount_20'):
        for row in read(QA/(name+'.json')):
            check(row['elements']>0 and row['max_pin_disp_mm']>0 and row['peak_vm_MPa']>0,'Bracket FEA')
    fine=read(QA/'mount_20.json')[0]['max_pin_disp_mm']
    coarse=read(QA/'mount_25.json')[0]['max_pin_disp_mm']
    check(abs(fine/coarse-1)<.05,'Bracket displacement mesh sensitivity')
    for name in ('frame_static','chassis_38','chassis_74'):
        check(read(QA/(name+'.json'))['region_peak_mm']['seat']>0,'Full chassis FEA')
    check(read(QA/'frame_static.json')['region_peak_mm']['seat']<read(QA/'before_braces.json')['region_peak_mm']['seat'],'Upper pivot stiffness improvement')
    check(read(QA/'seat_static.json')['peak_loaded_patch_down_mm']>0,'Seat FEA')
    if errors:raise SystemExit('Failed checks: '+', '.join(errors))
    print('Published geometry, dynamics and numerical regression checks passed.')
def rerun(cad,work):
    py=sys.executable
    def run(*args):subprocess.run(list(map(str,args)),check=True,cwd=HERE)
    model=HERE/'R6_0deg.FCStd'
    run(py,HERE/'make_scene.py')
    run(cad,QA/'pairs.py',model,QA/'all_pairs.json','--angles',*range(0,76,5))
    run(cad,HERE/'check_fit.py','--step',1)
    for script in ('joints','casters','slider'):run(cad,QA/(script+'.py'))
    run(cad,QA/'gears.py',model,QA/'gear_cycle.json')
    run(py,HERE/'stand.py','--suite')
    run(py,QA/'cycles.py')
    run(cad,QA/'frame_shape.py')
    for angle,stem in [(None,'frame_static'),(38,'chassis_38'),(74,'chassis_74')]:
        source=QA/('frame_loads.json' if angle is None else f'frame_loads_{angle}.json')
        extra=[] if angle is None else ['--angle',angle]
        run(py,QA/'frame_loads.py','--output',source,*extra)
        run(py,QA/'frame_fem.py',work/'frame','--loads',source,'--report',QA/(stem+'.json'),*(['--reuse-mesh'] if angle else []))
    comparison=work/'before_braces'
    comparison.mkdir()
    with gzip.open(QA/'before_braces_mesh.inp.gz','rb') as source, (comparison/'mesh.inp').open('wb') as dest:
        shutil.copyfileobj(source,dest)
    run(py,QA/'frame_fem.py',comparison,'--reuse-mesh','--report',QA/'before_braces.json')
    for h,stem,angles in [(2.5,'mount_25',[0,38,75]),(2,'mount_20',[0])]:
        run(py,QA/'mount_fem.py',work/stem,QA/(stem+'.json'),'--size',h,'--angles',*angles)
    run(py,QA/'seat_fem.py','seat','--size',4,'--work',work)
    (QA/'seat_static.json').write_bytes((work/'seat_4/report.json').read_bytes())
if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reports-only',action='store_true')
    cli.add_argument('--cad-python',default='/Applications/FreeCAD.app/Contents/Resources/bin/python')
    args=cli.parse_args()
    if not args.reports_only:
        with tempfile.TemporaryDirectory() as path:rerun(args.cad_python,Path(path))
    accept()
