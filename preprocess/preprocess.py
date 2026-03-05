from typing import List, Dict
import os
import cv2
import numpy as np
from PIL import Image

try:
    from facenet_pytorch import MTCNN
except Exception:
    MTCNN = None


def get_detector(device: str = 'cpu'):
    if MTCNN is None:
        raise RuntimeError('facenet-pytorch is required for face detection. Install from requirements.')
    return MTCNN(device=device, select_largest=True, post_process=False)


def detect_faces_pil(img: Image.Image, detector) -> List[Dict]:
    boxes, probs, landmarks = detector.detect(img, landmarks=True)
    results = []
    if boxes is None:
        return results
    for b, lm in zip(boxes, landmarks):
        x1, y1, x2, y2 = b
        w = x2 - x1
        h = y2 - y1
        results.append({'box': (int(x1), int(y1), int(w), int(h)), 'landmarks': np.array(lm)})
    return results


def align_and_crop(img: np.ndarray, landmarks: np.ndarray, size: int = 256) -> np.ndarray:
    dst = np.array([
        [0.3 * size, 0.35 * size],
        [0.7 * size, 0.35 * size],
        [0.5 * size, 0.55 * size],
        [0.35 * size, 0.75 * size],
        [0.65 * size, 0.75 * size],
    ], dtype=np.float32)

    src = landmarks.astype(np.float32)
    tfm = cv2.estimateAffinePartial2D(src, dst)[0]
    if tfm is None:
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        half = min(h, w) // 2
        crop = img[cy - half:cy + half, cx - half:cx + half]
        crop = cv2.resize(crop, (size, size))
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    warped = cv2.warpAffine(img, tfm, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)


def save_image_np(img_np: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = Image.fromarray(img_np.astype(np.uint8))
    img.save(out_path)
