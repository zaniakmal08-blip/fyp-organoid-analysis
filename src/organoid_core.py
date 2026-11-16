"""Shared local prototype: TIFF -> segmentation -> ImageJ measurements."""
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch
from train_three import TinyUNet

HERE = Path(__file__).resolve().parent
from project_paths import FIJI, MODEL, RESULTS
FIELDS = ['Area', 'Perimeter', 'Circularity', 'AspectRatio', 'Roundness', 'Solidity', 'Feret', 'MinFeret']
torch.set_num_threads(2)

def load_tiff(source):
    with Image.open(source) as im:
        if im.format != 'TIFF':
            raise ValueError('Please upload a TIFF image (.tif or .tiff).')
        if getattr(im, 'n_frames', 1) != 1:
            raise ValueError('This prototype accepts single-plane TIFFs. Export one agreed plane from a stack first.')
        if im.width * im.height > 30_000_000:
            raise ValueError('Image is larger than the prototype limit (30 megapixels).')
        a = np.asarray(im).copy()
        description = str(im.tag_v2.get(270, ''))
    if a.ndim == 3 and a.shape[2] in (3, 4):
        a = a[..., :3].mean(axis=-1)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('Expected a finite grayscale or RGB image.')
    a = a.astype(np.float32)
    if np.ptp(a) == 0:
        raise ValueError('Image has no intensity variation; cannot predict a useful boundary.')
    scale = None
    try:
        tree = ET.fromstring(description)
        pixels = next(p for p in tree.iter() if p.tag.split('}')[-1] == 'Pixels')
        def microns(axis):
            value = float(pixels.attrib['PhysicalSize'+axis])
            unit = pixels.attrib.get('PhysicalSize'+axis+'Unit', '\u00b5m')
            factors = {'\u00b5m':1, '\u03bcm':1, 'um':1, 'nm':0.001, 'mm':1000, 'm':1e6}
            return value * factors[unit]
        sx, sy = microns('X'), microns('Y')
        if math.isfinite(sx) and math.isfinite(sy) and sx > 0 and sy > 0:
            scale = {'x':sx, 'y':sy, 'source':'OME PhysicalSizeX/Y; requires operator confirmation'}
            if max(sx,sy)>100:
                scale['requires_manual_entry']=True
                scale['warning']='OME physical pixel sizes are unusually large for these microscope images and may represent field dimensions. Enter a verified pixel calibration; values were not converted automatically.'
    except (ValueError, KeyError, StopIteration, ET.ParseError):
        pass
    return a, scale

def normalize(a):
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a-lo)/max(float(hi-lo), 1e-6), 0, 1).astype(np.float32)

def letterbox(a, size, mask=False):
    h,w = a.shape
    ratio = size / max(h,w)
    nh,nw = max(1,round(h*ratio)),max(1,round(w*ratio))
    interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR
    resized = cv2.resize(a, (nw,nh), interpolation=interpolation)
    top,left = (size-nh)//2,(size-nw)//2
    result = np.zeros((size,size),np.float32)
    result[top:top+nh,left:left+nw] = resized
    return result, (top,left,nh,nw)

def imagej_paths():
    jars = sorted((FIJI/'jars').glob('ij-*.jar'))
    javas = list((FIJI/'java').rglob('bin/java.exe'))
    if not jars or not javas:
        raise RuntimeError('Fiji installation not found. Set the FIJI_HOME environment variable to your Fiji directory (including bundled Java).')
    return jars[-1],javas[0]

def measure_imagej(mask_path, output_dir, sx, sy, roi_input=None):
    jar,java = imagej_paths()
    helper = HERE/'imagej_helper'/'MeasureOrganoid.java'
    command=[str(java),'-Djava.awt.headless=true','-cp',str(jar),str(helper),
             str(Path(mask_path).resolve()), str(Path(output_dir).resolve()), str(sx),str(sy)]
    if roi_input: command.append(str(Path(roi_input).resolve()))
    result=subprocess.run(command,capture_output=True,text=True,timeout=90)
    if result.returncode:
        raise RuntimeError('ImageJ measurement failed: '+result.stderr[-2000:])
    return json.loads((Path(output_dir)/'measurements.json').read_text())

def load_model(model_path=MODEL):
    model_path=Path(model_path)
    if not model_path.exists():
        raise RuntimeError('Train the prototype model first: run train_prototype.py.')
    checkpoint_bytes=model_path.read_bytes()
    checkpoint=torch.load(io.BytesIO(checkpoint_bytes),map_location='cpu',weights_only=True)
    checkpoint['_model_sha256']=hashlib.sha256(checkpoint_bytes).hexdigest()
    if checkpoint.get('preprocessing') != 'letterbox_percentile_v1':
        raise RuntimeError('This app requires the aspect-preserving prototype checkpoint.')
    checkpoint['_model_path']=str(model_path.resolve())
    model=TinyUNet(); model.load_state_dict(checkpoint['model_state']); model.eval()
    return model,checkpoint

def analyse(data, filename, output_dir, model, checkpoint, scale=None):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=False)
    a,metadata_scale=load_tiff(io.BytesIO(data))
    if scale is None:
        raise ValueError('Confirm the image scale before analysis.')
    sx,sy=float(scale['x']),float(scale['y'])
    if not all(math.isfinite(v) and 0<v<=1000 for v in (sx,sy)):
        raise ValueError('Scale must be positive, finite and no greater than 1000 um/pixel.')
    (output_dir/'input.tif').write_bytes(data)
    gray=normalize(a)
    boxed,(top,left,nh,nw)=letterbox(gray,checkpoint['size'])
    with torch.no_grad():
        prob=model(torch.from_numpy(boxed[None,None])).sigmoid().numpy()[0,0]
    prob=cv2.resize(prob[top:top+nh,left:left+nw],(a.shape[1],a.shape[0]),interpolation=cv2.INTER_LINEAR)
    raw=np.uint8(prob>=0.5)*255
    cv2.imwrite(str(output_dir/'raw_mask.png'),raw)
    np.save(output_dir/'probability.npy',prob)
    contours,_=cv2.findContours(raw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise ValueError('Model predicted an empty region; no measurements produced.')
    contour=max(contours,key=cv2.contourArea)
    if cv2.contourArea(contour)<16:
        raise ValueError('Predicted object is too small to measure.')
    mask=np.zeros_like(raw); cv2.drawContours(mask,[contour],-1,255,cv2.FILLED)
    internal_pixels_filled=int(np.count_nonzero((mask>0)&(raw==0)))
    # Fixed pixel-scale raster-to-polygon conversion, not a fitted circle/ellipse.
    # Old checkpoints retain their original measurement representation for comparison.
    roi_input=None
    representation=checkpoint.get('roi_representation','threshold_mask')
    if representation=='polygon_1px':
        contour=cv2.approxPolyDP(contour,1.0,True)
        if len(contour)<3: raise ValueError('Predicted polygon has fewer than three vertices.')
        roi_input=output_dir/'roi_vertices.csv'
        with roi_input.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['x','y'])
            writer.writerows((float(p[0])+0.5,float(p[1])+0.5) for p in contour[:,0])
        mask[:]=0;cv2.drawContours(mask,[contour],-1,255,cv2.FILLED)
    cv2.imwrite(str(output_dir/'mask.png'),mask)
    overlay=cv2.cvtColor(np.uint8(gray*255),cv2.COLOR_GRAY2RGB)
    cv2.drawContours(overlay,[contour],-1,(0,255,0),max(2,round(max(a.shape)/700)))
    Image.fromarray(overlay).save(output_dir/'overlay.png')
    Image.fromarray(np.uint8(gray*255)).save(output_dir/'input_preview.png')
    values=measure_imagej(output_dir/'mask.png',output_dir,sx,sy,roi_input)
    warnings=[f'{len(checkpoint["names"])}-image demonstration model: inspect the outline before using these measurements.',
              'Mask restored to original coordinates; boundary detail is limited by model input resolution.']
    if len(contours)>1: warnings.append(f'{len(contours)-1} smaller external regions discarded; only the largest object is measured.')
    if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():
        warnings.append('Predicted object touches the image edge; it may be incomplete.')
    # Hole filling makes the selected region represent the whole outer organoid.
    filled=internal_pixels_filled
    if filled: warnings.append(f'{filled} internal background pixels filled inside the selected outer boundary.')
    model_path=Path(checkpoint.get('_model_path',MODEL))
    result={'filename':filename,'measurements':values,'scale_um_per_pixel':{'x':sx,'y':sy},
            'scale_source':scale['source'],'detected_metadata_scale':metadata_scale,
            'width':a.shape[1],'height':a.shape[0],'warnings':warnings,
            'model':str(model_path),'training_images':checkpoint['names'],
            'input_sha256':hashlib.sha256(data).hexdigest(),
            'model_sha256':checkpoint['_model_sha256'],
            'measurement_engine':'ImageJ 1.54p; calibrated original image coordinates',
            'roi_representation':representation,
            'postprocessing':('largest external region; filled interior; polygon simplification tolerance 1 original pixel; pixel-centre coordinates' if representation=='polygon_1px' else 'largest external region, filled interior, no contour smoothing'),
            'model_input_size':checkpoint['size'],'review_status':'unreviewed',
            'units':{'Area':'um^2','Perimeter':'um','Feret':'um','MinFeret':'um',
                     'Circularity':'dimensionless','AspectRatio':'dimensionless','Roundness':'dimensionless','Solidity':'dimensionless'}}
    (output_dir/'result.json').write_text(json.dumps(result,indent=2))
    with (output_dir/'measurements.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['Image','Area_um2','Perimeter_um','Circularity','AspectRatio','Roundness','Solidity','Feret_um','MinFeret_um','ScaleX_um_per_pixel','ScaleY_um_per_pixel','ReviewStatus']
        writer=csv.writer(f); writer.writerow(fields)
        writer.writerow([filename]+[values[k] for k in FIELDS]+[sx,sy,'unreviewed'])
    with zipfile.ZipFile(output_dir/'results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in output_dir.iterdir():
            if p.is_file() and p.name != 'results.zip': z.write(p,p.name)
    return result
