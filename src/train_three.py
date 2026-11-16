"""Three-image U-Net memorisation experiment. No generalisation claim."""
from pathlib import Path
import argparse
import csv
import json
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

from project_paths import DATA as ROOT, RESULTS
NAMES = ['B01_P01_A01_H2O2_000uM_D01_10x',
         'B01_P01_A02_H2O2_12.5uM_D02_10x',
         'B01_P01_A03_H2O2_025uM_D04_10x']

def prepare_image(path, size):
    a = np.asarray(Image.open(path)).astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1)
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip((a-lo) / max(float(hi-lo), 1e-6), 0, 1)
    return np.asarray(Image.fromarray(a).resize((size, size), Image.Resampling.BILINEAR)).copy()

def block(a, b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),
                         nn.Conv2d(b,b,3,padding=1),nn.ReLU())

class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1=block(1,8); self.e2=block(8,16); self.middle=block(16,32)
        self.pool=nn.MaxPool2d(2)
        self.u2=nn.ConvTranspose2d(32,16,2,stride=2); self.d2=block(32,16)
        self.u1=nn.ConvTranspose2d(16,8,2,stride=2); self.d1=block(16,8)
        self.out=nn.Conv2d(8,1,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(self.pool(a)); c=self.middle(self.pool(b))
        d=self.d2(torch.cat([self.u2(c),b],dim=1))
        return self.out(self.d1(torch.cat([self.u1(d),a],dim=1)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--size',type=int,default=256)
    args=parser.parse_args()
    if args.epochs < 1 or args.size < 16 or args.size % 4:
        parser.error('epochs must be positive; size must be >=16 and divisible by 4')
    missing=[str(ROOT/folder/(n+ext)) for n in NAMES
             for folder,ext in [('images','.tif'),('masks','.png')]
             if not (ROOT/folder/(n+ext)).exists()]
    if missing:
        raise SystemExit('Missing image/mask files. Restore them before training:\n'+'\n'.join(missing))
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(2)
    xs=[]; ys=[]
    for n in NAMES:
        ip=ROOT/'images'/(n+'.tif'); mp=ROOT/'masks'/(n+'.png')
        with Image.open(ip) as im: shape=im.size
        with Image.open(mp) as im:
            m=np.asarray(im)
            if im.size != shape or m.ndim != 2 or not set(np.unique(m)).issubset({0,255}) or len(np.unique(m)) != 2:
                raise SystemExit('Mask must match image dimensions and contain both 0 and 255: '+str(mp))
            mask=np.asarray(im.resize((args.size,args.size),Image.Resampling.NEAREST)).copy()/255.
        xs.append(prepare_image(ip,args.size)); ys.append(mask)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x=torch.tensor(np.array(xs)[:,None],dtype=torch.float32,device=device)
    y=torch.tensor(np.array(ys)[:,None],dtype=torch.float32,device=device)
    model=TinyUNet().to(device); optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    output=RESULTS/'baseline_runs'
    from datetime import datetime
    output=output/datetime.now().strftime('%Y%m%d_%H%M%S'); output.mkdir(parents=True)
    print('Training on all 3 pairs; no validation/test set. Device:',device,flush=True)
    history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); optimizer.zero_grad(); logits=model(x); prob=logits.sigmoid()
        dims=(1,2,3)
        dice=(2*(prob*y).sum(dims)+1)/(prob.sum(dims)+y.sum(dims)+1)
        loss=nn.functional.binary_cross_entropy_with_logits(logits,y)+(1-dice).mean()
        loss.backward(); optimizer.step()
        history.append({'epoch':epoch,'training_loss':loss.item()})
        if epoch==1 or epoch%20==0:
            print(f'Epoch {epoch}/{args.epochs}: training loss {loss.item():.4f}',flush=True)
    model.eval()
    with torch.no_grad(): probs=model(x).sigmoid().cpu().numpy()[:,0]
    metrics=[]
    for i,n in enumerate(NAMES):
        pred=probs[i]>=0.5; truth=np.array(ys[i])>0
        intersection=(pred&truth).sum(); union=(pred|truth).sum()
        metrics.append({'image':n,'training_IoU':float(intersection/max(union,1)),
                        'training_Dice':float(2*intersection/max(pred.sum()+truth.sum(),1))})
        Image.fromarray(np.uint8(pred)*255).save(output/(n+'_prediction_256.png'))
        panels=Image.new('RGB',(args.size*3,args.size+30),'white'); draw=ImageDraw.Draw(panels)
        for j,(a,label) in enumerate([(xs[i],'Input'),(ys[i],'Manual mask'),(pred,'TRAINING prediction')]):
            panels.paste(Image.fromarray(np.uint8(a*255)).convert('RGB'),(j*args.size,30))
            draw.text((j*args.size+5,5),label,fill='black')
        panels.save(output/(n+'_comparison.png'))
    torch.save({'model_state':model.cpu().state_dict(),'size':args.size,
                'preprocessing':'grayscale; per-image 1-99 percentile clip; bilinear resize',
                'names':NAMES,'seed':42,'experiment':'memorisation_all_three_no_holdout'},output/'tiny_unet.pt')
    (output/'training_metrics.json').write_text(json.dumps(metrics,indent=2))
    with (output/'training_history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['epoch','training_loss']);w.writeheader();w.writerows(history)
    print(json.dumps(metrics,indent=2));print('Saved outputs:',output)
    print('These are training-image results, not unseen-image accuracy. Preview masks are resized; do not use them for calibrated morphometry.')

if __name__=='__main__':
    main()
