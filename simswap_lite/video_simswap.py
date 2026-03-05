import argparse
from pathlib import Path

import cv2
import torch
from facenet_pytorch import MTCNN

from infer_simswap import (
    load_id_encoder,
    load_image_bgr,
    load_swap_generator,
    swap_single_face,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--id_ckpt', required=True)
    parser.add_argument('--swap_ckpt', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--smooth_alpha', type=float, default=0.7, help='EMA blend for temporal smoothing')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    id_model, id_tfm, _ = load_id_encoder(args.id_ckpt, device)
    gen, gen_tfm, gen_size = load_swap_generator(args.swap_ckpt, device)
    source_bgr = load_image_bgr(args.source)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )

    frame_count = 0
    prev_frame = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames is not None and frame_count >= args.max_frames:
            break

        try:
            swapped = swap_single_face(source_bgr, frame, mtcnn, id_model, id_tfm, gen, gen_tfm, gen_size)
        except Exception:
            swapped = frame

        if prev_frame is not None:
            swapped = cv2.addWeighted(swapped, 1.0 - args.smooth_alpha, prev_frame, args.smooth_alpha, 0.0)

        writer.write(swapped)
        prev_frame = swapped.copy()
        frame_count += 1

    cap.release()
    writer.release()
    print(f'Saved video to {out_path} ({frame_count} frames)')


if __name__ == '__main__':
    main()
