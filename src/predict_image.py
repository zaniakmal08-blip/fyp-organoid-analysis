"""Run the saved three-image model. Press any key in the OpenCV window to close."""
from pathlib import Path
import argparse
import cv2
import numpy as np
from PIL import Image
import torch
from train_three import TinyUNet, prepare_image

# EDIT THIS PATH, then run this script. A04 was not used during training.
INPUT_IMAGE = Path(r'c:\Users\User\Desktop\MachineVision\MCTA-4364-Machine-Vision\notebooks\images\syazani.jpg')
MODEL_FILE = Path(__file__).parent / 'runs' / '20261022_172316' / 'tiny_unet.pt'
THRESHOLD = 0.5

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=INPUT_IMAGE)
    parser.add_argument('--model', type=Path, default=MODEL_FILE)
    parser.add_argument('--no-show', action='store_true', help='Save preview without opening a window')
    args = parser.parse_args()
    if not args.image.is_file():
        raise SystemExit(f'Image not found: {args.image}')
    if not args.model.is_file():
        raise SystemExit(f'Model not found: {args.model}')
    torch.set_num_threads(2)
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
    model = TinyUNet()
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    size = checkpoint['size']
    image = prepare_image(args.image, size)
    x = torch.tensor(image[None, None], dtype=torch.float32)
    with torch.no_grad():
        probability = model(x).sigmoid().numpy()[0, 0]
    mask = np.uint8(probability >= THRESHOLD) * 255
    # Undo the preview resize for display alignment only: no new detail is recovered.
    with Image.open(args.image) as original:
        width, height = original.size
        a = np.asarray(original).astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1)
    low, high = np.percentile(a, [1, 99])
    gray = np.uint8(np.clip((a-low)/max(float(high-low), 1e-6), 0, 1)*255)
    display_size = (480, max(1, round(480*height/width)))
    source = cv2.cvtColor(cv2.resize(gray, display_size), cv2.COLOR_GRAY2BGR)
    shown_mask = cv2.resize(mask, display_size, interpolation=cv2.INTER_NEAREST)
    overlay = source.copy()
    contours, _ = cv2.findContours(shown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    panels = []
    for content, title in [(source, 'Input image'),
                           (cv2.cvtColor(shown_mask, cv2.COLOR_GRAY2BGR), 'AI mask'),
                           (overlay, 'AI outline (unreviewed)')]:
        panel = cv2.copyMakeBorder(content, 45, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25,25,25))
        cv2.putText(panel, title, (12,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
        panels.append(panel)
    preview = np.hstack(panels)
    out = Path(__file__).parent/'predictions'
    out.mkdir(exist_ok=True)
    preview_path = out/(args.image.stem+'_preview.png')
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError('Could not save preview')
    cv2.imwrite(str(out/(args.image.stem+f'_mask_{size}px.png')), mask)
    print('Input:', args.image)
    print('Preview:', preview_path)
    print('Model trained on 3 images; prediction requires review.')
    print('Mask is a resized demonstration, not a calibrated measurement output.')
    if not args.no_show:
        window = 'Organoid AI | Input - Mask - Outline | Press any key to close'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, preview)
        cv2.resizeWindow(window, 1200, 400)
        while cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(100) >= 0:
                break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
