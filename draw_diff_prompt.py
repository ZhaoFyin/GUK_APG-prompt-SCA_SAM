import json
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


mask_dir = r"G:\KGE-SwinFpn/VOCdevkit_YYL\VOC_landslide\SegmentationObject"
rgb_dir = r"G:\KGE-SwinFpn/VOCdevkit_YYL\VOC_landslide\JPEGImages"
prompt_dir = "./prompt"
tmp_dir = "./tmp"
classes = ['非滑坡', '滑坡']
PALETTE = [(0,  0,  0), (255, 255,  255)]
idx = 30


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



def to_binary_mask_512(mask):
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    if mask.ndim == 2:
        return (mask != 0).astype(np.uint8)
    return (np.any(mask != 0, axis=2)).astype(np.uint8)


def draw_legend(image):
    entries = [
        ("质心", (0, 255, 255)),
        ("偏移质心", (255, 255, 0)),
        ("外接框", (0, 255, 0)),
        ("尺寸偏移框", (255, 0, 0)),
        ("位置偏移框", (0, 0, 255)),
    ]

    left = 10
    top = 10
    line_h = 28
    panel_w = 150
    panel_h = 14 + len(entries) * line_h

    # 先画白底面板
    cv2.rectangle(image, (left, top), (left + panel_w, top + panel_h), (255, 255, 255), -1)
    cv2.rectangle(image, (left, top), (left + panel_w, top + panel_h), (0, 0, 0), 1)

    # 先画色块
    for i, (name, color) in enumerate(entries):
        y = top + 20 + i * line_h
        cv2.rectangle(image, (left + 10, y - 8), (left + 26, y + 8), color, -1)

    # 用 PIL 写中文
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Windows 常用中文字体，按你的机器实际情况修改
    font_path = r"C:\Windows\Fonts\simsun.ttc"
    font = ImageFont.truetype(font_path, 18)

    for i, (name, color) in enumerate(entries):
        y = top + 20 + i * line_h
        draw.text((left + 34, y - 10), name, font=font, fill=(0, 0, 0))

    # 转回 OpenCV BGR
    image[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def vision_check(sample_id):
    os.makedirs(tmp_dir, exist_ok=True)
    if not os.path.isdir(prompt_dir):
        return

    json_files = sorted([f for f in os.listdir(prompt_dir) if f.lower().endswith(".json")])
    json_name = json_files[sample_id]

    json_path = os.path.join(prompt_dir, json_name)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_name = data.get("file_name")

    file_path = os.path.join(mask_dir, file_name)
    rgb_path = os.path.join(rgb_dir, file_name.split(".")[0] + ".jpg")

    mask = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_NEAREST)

    mask01 = to_binary_mask_512(mask)
    canvas = overlap(rgb, mask01, mask01)
    vis = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    box = data.get("box", None)
    box_size = data.get("box_with_size_offset", None)
    box_pos = data.get("box_with_position_offset", None)
    c1 = data.get("centroid", None)
    c2 = data.get("centroid_with_offset", None)

    if box is not None and len(box) == 4:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if box_size is not None and len(box_size) == 4:
        x1, y1, x2, y2 = [int(round(v)) for v in box_size]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    if box_pos is not None and len(box_pos) == 4:
        x1, y1, x2, y2 = [int(round(v)) for v in box_pos]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if c1 is not None and len(c1) == 2:
        x, y = int(round(c1[0])), int(round(c1[1]))
        cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)
        cv2.circle(vis, (x, y), 6, (0, 0, 0), 1)
    if c2 is not None and len(c2) == 2:
        x, y = int(round(c2[0])), int(round(c2[1]))
        cv2.circle(vis, (x, y), 4, (255, 255, 0), -1)
        cv2.circle(vis, (x, y), 6, (0, 0, 0), 1)

    draw_legend(vis)

    out_name = "./fig/Diff_pmt.png"

    cv2.imwrite(out_name, vis)

if __name__ == "__main__":
    vision_check(sample_id=idx)