import argparse
import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms

from simswap_style_models import IdentityEncoder, SimSwapLiteGenerator


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
LOGGER = logging.getLogger('infer_simswap')

REFERENCE_5PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def get_primary_face(mtcnn: MTCNN, image_bgr: np.ndarray):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = mtcnn.detect(Image.fromarray(rgb), landmarks=True)
    if boxes is None or len(boxes) == 0:
        raise RuntimeError('No face detected.')
    idx = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
    return boxes[idx], landmarks[idx]


def estimate_affine_5pt(src_landmarks: np.ndarray, dst_landmarks: np.ndarray) -> np.ndarray:
    M, _ = cv2.estimateAffinePartial2D(src_landmarks.astype(np.float32), dst_landmarks.astype(np.float32), method=cv2.LMEDS)
    if M is None:
        raise RuntimeError('Affine estimation failed.')
    return M


def align_face(image_bgr: np.ndarray, landmarks_5: np.ndarray, out_size: int = 224):
    dst = REFERENCE_5PTS.copy() * (out_size / 112.0)
    M = estimate_affine_5pt(landmarks_5, dst)
    aligned = cv2.warpAffine(image_bgr, M, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return aligned, M


def inverse_warp(src_aligned: np.ndarray, target_shape: Tuple[int, int], M_target: np.ndarray) -> np.ndarray:
    M3 = np.vstack([M_target, [0, 0, 1]])
    M_inv = np.linalg.inv(M3)[:2]
    h, w = target_shape
    return cv2.warpAffine(src_aligned, M_inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def make_soft_face_mask(shape_hw: Tuple[int, int], landmarks_5: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    center_x = int(np.mean(landmarks_5[:, 0]))
    center_y = int(np.mean(landmarks_5[:, 1]) + 8)
    face_w = int(max(70, 2.2 * np.linalg.norm(landmarks_5[1] - landmarks_5[0])))
    face_h = int(face_w * 1.35)
    cv2.ellipse(mask, (center_x, center_y), (face_w // 2, face_h // 2), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    return mask


def color_correct(source_face: np.ndarray, target_region: np.ndarray, mask: np.ndarray) -> np.ndarray:
    src_lab = cv2.cvtColor(source_face, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = mask > 0
    if m.sum() < 10:
        return source_face
    for c in range(3):
        src_vals = src_lab[:, :, c][m]
        tgt_vals = tgt_lab[:, :, c][m]
        src_mean, src_std = float(src_vals.mean()), float(src_vals.std() + 1e-6)
        tgt_mean, tgt_std = float(tgt_vals.mean()), float(tgt_vals.std() + 1e-6)
        src_lab[:, :, c] = ((src_lab[:, :, c] - src_mean) * (tgt_std / src_std)) + tgt_mean
    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)


def blend_face(warped_face: np.ndarray, target_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    warped_face: HxWx3 uint8, same size as target_bgr
    target_bgr:  HxWx3 uint8
    mask:        HxW or HxWx1/3, uint8 or float
    """
    h, w = target_bgr.shape[:2]

    # Make sure face image matches target canvas
    if warped_face.shape[:2] != (h, w):
        warped_face = cv2.resize(warped_face, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize mask to single-channel uint8 [0,255]
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    if mask.dtype != np.uint8:
        if mask.max() <= 1.0:
            mask = (mask * 255.0).clip(0, 255).astype(np.uint8)
        else:
            mask = mask.clip(0, 255).astype(np.uint8)

    # If mask is empty, just return target
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return target_bgr.copy()

    # Compute ROI center from actual nonzero mask area
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    roi_w = x1 - x0 + 1
    roi_h = y1 - y0 + 1

    center_x = x0 + roi_w // 2
    center_y = y0 + roi_h // 2

    # Clamp center so ROI fits inside target image
    half_w = roi_w // 2
    half_h = roi_h // 2

    center_x = max(half_w, min(w - half_w - 1, center_x))
    center_y = max(half_h, min(h - half_h - 1, center_y))

    center = (int(center_x), int(center_y))

    try:
        return cv2.seamlessClone(
            warped_face,
            target_bgr,
            mask,
            center,
            cv2.NORMAL_CLONE
        )
    except cv2.error:
        # Safe fallback: alpha blend
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        out = warped_face.astype(np.float32) * alpha + target_bgr.astype(np.float32) * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

@torch.no_grad()
def load_id_encoder(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = IdentityEncoder(num_classes=ckpt['num_classes'], embedding_dim=ckpt.get('embedding_dim', 256))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((ckpt.get('image_size', 160), ckpt.get('image_size', 160))),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return model, tfm, ckpt.get('embedding_dim', 256)


@torch.no_grad()
def load_swap_generator(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SimSwapLiteGenerator(style_dim=ckpt['style_dim'], base_channels=ckpt.get('base_channels', 64))
    model.load_state_dict(ckpt['generator_state_dict'])
    model.to(device).eval()
    image_size = ckpt.get('image_size', 224)
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return model, tfm, image_size


@torch.no_grad()
def get_id_embedding(id_model, id_tfm, device, bgr_img: np.ndarray) -> torch.Tensor:
    x = id_tfm(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    emb, _ = id_model(x)
    return emb


@torch.no_grad()
def run_generator(gen, gen_tfm, device, target_aligned_bgr: np.ndarray, src_id_emb: torch.Tensor) -> np.ndarray:
    x = gen_tfm(cv2.cvtColor(target_aligned_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    fake = gen(x, src_id_emb)
    fake = ((fake.squeeze(0).clamp(-1, 1) + 1.0) / 2.0).cpu().permute(1, 2, 0).numpy()
    fake = (fake * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(fake, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def embedding_similarity(id_model, id_tfm, device, img1_bgr, img2_bgr) -> float:
    e1 = get_id_embedding(id_model, id_tfm, device, img1_bgr)
    e2 = get_id_embedding(id_model, id_tfm, device, img2_bgr)
    return float(F.cosine_similarity(e1, e2).item())


def swap_single_face(source_bgr: np.ndarray, target_bgr: np.ndarray, mtcnn: MTCNN, id_model, id_tfm, gen, gen_tfm, gen_size: int):
    _, src_lm = get_primary_face(mtcnn, source_bgr)
    _, tgt_lm = get_primary_face(mtcnn, target_bgr)

    src_aligned, _ = align_face(source_bgr, src_lm, out_size=gen_size)
    tgt_aligned, M_tgt = align_face(target_bgr, tgt_lm, out_size=gen_size)

    src_id_emb = get_id_embedding(id_model, id_tfm, mtcnn.device, src_aligned)
    fake_aligned = run_generator(gen, gen_tfm, mtcnn.device, tgt_aligned, src_id_emb)
    warped_fake = inverse_warp(fake_aligned, target_bgr.shape[:2], M_tgt)

    mask = make_soft_face_mask(target_bgr.shape[:2], tgt_lm)
    corrected = color_correct(warped_fake, target_bgr, mask)
    swapped = blend_face(corrected, target_bgr, mask)
    return swapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--id_ckpt', required=True)
    parser.add_argument('--swap_ckpt', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    id_model, id_tfm, _ = load_id_encoder(args.id_ckpt, device)
    gen, gen_tfm, gen_size = load_swap_generator(args.swap_ckpt, device)

    source_bgr = load_image_bgr(args.source)
    target_bgr = load_image_bgr(args.target)
    swapped = swap_single_face(source_bgr, target_bgr, mtcnn, id_model, id_tfm, gen, gen_tfm, gen_size)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), swapped):
        raise RuntimeError(f'Failed to save: {out}')

    sim = embedding_similarity(id_model, id_tfm, device, source_bgr, swapped)
    LOGGER.info('Identity similarity (source vs swapped): %.4f', sim)
    LOGGER.info('Saved swapped image to %s', out)


if __name__ == '__main__':
    main()
