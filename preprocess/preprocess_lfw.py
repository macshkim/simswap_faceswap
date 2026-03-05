import csv
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np

from src.preprocess import get_detector, detect_faces_pil, align_and_crop, save_image_np


def process_lfw(lfw_dir: str, out_dir: str, size: int = 256, device: str = 'cpu', min_imgs: int = 1):
    detector = get_detector(device=device)
    image_paths = glob(os.path.join(lfw_dir, '**', '*.jpg'), recursive=True)
    metadata = []

    for p in tqdm(image_paths, desc='Images'):
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            continue
        results = detect_faces_pil(img, detector)
        if not results:
            continue
        # take largest (detector configured to select largest)
        r = results[0]
        img_np = np.array(img)[:, :, ::-1].copy()  # PIL RGB -> BGR for opencv routines
        crop = align_and_crop(img_np, r['landmarks'], size=size)
        rel = os.path.relpath(p, lfw_dir)
        out_path = os.path.join(out_dir, rel)
        out_path = os.path.splitext(out_path)[0] + '.png'
        save_image_np(crop, out_path)
        identity = os.path.basename(os.path.dirname(p))
        metadata.append({'image_path': out_path, 'identity': identity})

    # Write metadata CSV
    csv_path = os.path.join(out_dir, 'metadata.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'identity'])
        writer.writeheader()
        for row in metadata:
            writer.writerow(row)
    print(f'Wrote metadata to {csv_path} with {len(metadata)} images')


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--lfw-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--min-imgs', type=int, default=1)
    args = parser.parse_args()

    process_lfw(args.lfw_dir, args.out_dir, size=args.size, device=args.device, min_imgs=args.min_imgs)

