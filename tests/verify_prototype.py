import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from project_paths import RESULTS
"""Integration checks: manual Fiji agreement, AI exports, and invalid inputs."""
import csv, io, json, tempfile, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from organoid_core import HERE,FIELDS,load_tiff,load_model,analyse,measure_imagej
from train_three import ROOT,NAMES

def main():
    scale=0.61922571983322461
    model,checkpoint=load_model()
    out=RESULTS/'prototype_checks'; out.mkdir(exist_ok=True)
    keymap={'Area':'Area','Perimeter':'Perim.','Circularity':'Circ.','AspectRatio':'AR',
            'Roundness':'Round','Solidity':'Solidity','Feret':'Feret','MinFeret':'MinFeret'}
    report={'manual_roi_agreement':[],'prediction_exports':[],'rejections':[]}
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        tmp=Path(tmp)
        for name in NAMES:
            manual=measure_imagej(ROOT/'masks'/(name+'.png'),tmp/(name+'_reference'),scale,scale,ROOT/'rois'/(name+'.roi'))
            with (ROOT/'measurements'/(name+'.csv')).open() as f: reference=next(csv.DictReader(f))
            errors={k:abs(manual[k]-float(reference[keymap[k]])) for k in FIELDS}
            for k in FIELDS:
                assert errors[k] <= max(1e-6,abs(float(reference[keymap[k]]))*1e-6),(name,k,errors[k])
            report['manual_roi_agreement'].append({'image':name,'absolute_differences':errors})
            dest=tmp/(name+'_prediction')
            r=analyse((ROOT/'images'/(name+'.tif')).read_bytes(),name+'.tif',dest,model,checkpoint,{'x':scale,'y':scale,'source':'integration test pilot scale'})
            assert set(r['measurements'])==set(FIELDS)
            assert Image.open(dest/'mask.png').size==Image.open(ROOT/'images'/(name+'.tif')).size
            assert (dest/'predicted.roi').read_bytes()[:4]==b'Iout'
            with zipfile.ZipFile(dest/'results.zip') as z: assert z.testzip() is None and 'measurements.csv' in z.namelist()
            with (dest/'measurements.csv').open(encoding='utf-8-sig') as f:
                row=next(csv.DictReader(f));assert abs(float(row['Area_um2'])-r['measurements']['Area'])<1e-6
            Image.open(dest/'overlay.png').save(out/(name+'_predicted_overlay.png'))
            report['prediction_exports'].append({'image':name,'measurements':r['measurements'],'warnings':r['warnings']})
        # Physical scale must change length by two and area by four.
        name=NAMES[0]
        original=measure_imagej(ROOT/'masks'/(name+'.png'),tmp/'scale1',1,1)
        doubled=measure_imagej(ROOT/'masks'/(name+'.png'),tmp/'scale2',2,2)
        assert abs(doubled['Area']/original['Area']-4)<1e-9
        assert abs(doubled['Feret']/original['Feret']-2)<1e-9
        for k in ['Circularity','AspectRatio','Roundness','Solidity']: assert abs(original[k]-doubled[k])<1e-9
        for typ in ['blank','stack','png']:
            buff=io.BytesIO()
            im=Image.fromarray(np.arange(1024,dtype=np.uint8).reshape(32,32))
            if typ=='blank': Image.new('L',(32,32),0).save(buff,format='TIFF')
            elif typ=='stack': im.save(buff,format='TIFF',save_all=True,append_images=[im])
            else: im.save(buff,format='PNG')
            try: load_tiff(io.BytesIO(buff.getvalue()))
            except ValueError: report['rejections'].append(typ)
            else: raise AssertionError('Invalid input accepted: '+typ)
    report['scale_check']='passed: area x4, length x2, dimensionless unchanged'
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
