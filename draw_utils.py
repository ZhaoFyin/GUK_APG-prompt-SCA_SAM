import numpy as np
import cv2
import re
from matplotlib.font_manager import FontProperties

classes = ['非滑坡', '滑坡']
PALETTE = [(0,  0,  0), (255, 0,  0)]


def overlap(img, pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    overlay = img.copy()

    unique_classes = np.unique(pred)

    for cls_idx in unique_classes:
        if cls_idx >= len(classes): continue  # 忽略非法索引
        color = PALETTE[int(cls_idx)]
        mask = (pred == cls_idx).astype(np.uint8)
        color_layer = np.zeros_like(img, dtype=np.uint8)
        color_layer[mask == 1] = color

        # 仅在mask区域混合颜色
        # 逻辑：img[mask] = 0.6*img[mask] + 0.4*color
        roi = overlay[mask == 1]
        blended = cv2.addWeighted(roi, 0.3, color_layer[mask == 1], 0.7, 0)
        overlay[mask == 1] = blended

        # 加粗边界
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness=2)

    if 255 in gt:
        mask_255 = (gt == 255).astype(np.uint8)

        # 将255区域涂黑 (完全覆盖)
        overlay[mask_255 == 1] = [0, 0, 0]

        # 画白色边框区分黑色区域 (可选)
        contours_255, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_255, -1, (255, 255, 255), thickness=1)
    return overlay


def diff_color(s):
    s = np.asarray(s)
    if s.ndim == 3:
        s = s[:, :, 0]

    out = np.zeros((s.shape[0], s.shape[1], 3), dtype=np.uint8)
    unique_vals = np.unique(s)
    rng = np.random.default_rng()

    for val in unique_vals:
        if val == 0:
            continue
        color = rng.integers(0, 256, size=3, dtype=np.uint8)
        out[s == val] = color

    return out

