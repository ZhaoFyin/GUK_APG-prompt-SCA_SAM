# box point generator

import json
import os

import cv2
import numpy as np


mask_dir = r"G:\KGE-SwinFpn\VOCdevkit_YYL\VOC_landslide\SegmentationObject"
seed = 1234
prompt_dir = "./prompt"
tmp_dir = "./tmp"


def clamp(v, low, high):
    return max(low, min(high, v))


def clip_box_xyxy(x1, y1, x2, y2, width=512, height=512):
    x1 = clamp(float(x1), 0.0, float(width - 1))
    y1 = clamp(float(y1), 0.0, float(height - 1))
    x2 = clamp(float(x2), 0.0, float(width - 1))
    y2 = clamp(float(y2), 0.0, float(height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def box_from_center_size(cx, cy, bw, bh, width=512, height=512):
    bw = clamp(float(bw), 1.0, float(width))
    bh = clamp(float(bh), 1.0, float(height))
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return clip_box_xyxy(x1, y1, x2, y2, width=width, height=height)


def get_largest_component(mask01):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if num_labels <= 1:
        return None
    # label 0 is background; choose the largest foreground component.
    largest_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (labels == largest_label).astype(np.uint8)


def nearest_fg_point(x, y, xs, ys):
    d2 = (xs - x) * (xs - x) + (ys - y) * (ys - y)
    idx = int(np.argmin(d2))
    return int(xs[idx]), int(ys[idx])


def get_fg_candidates_within_offset(cx, cy, xs, ys, box_w, box_h):
    dx_limit = 0.3 * box_w
    dy_limit = 0.3 * box_h
    keep = (np.abs(xs - cx) <= dx_limit) & (np.abs(ys - cy) <= dy_limit)
    cands_x = xs[keep]
    cands_y = ys[keep]
    # centroid point itself is on foreground, so candidates should never be empty.
    if cands_x.size == 0:
        return np.array([cx], dtype=np.int32), np.array([cy], dtype=np.int32)
    return cands_x, cands_y


def to_binary_mask_512(mask):
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    if mask.ndim == 2:
        return (mask != 0).astype(np.uint8)
    return (np.any(mask != 0, axis=2)).astype(np.uint8)


def draw_legend(image):
    entries = [
        ("centroid", (0, 255, 255)),
        ("centroid_with_offset", (255, 255, 0)),
        ("box", (0, 255, 0)),
        ("box_with_size_offset", (255, 0, 0)),
        ("box_with_position_offset", (0, 0, 255)),
    ]

    left = 10
    top = 10
    line_h = 24
    panel_w = 290
    panel_h = 14 + len(entries) * line_h

    cv2.rectangle(image, (left, top), (left + panel_w, top + panel_h), (255, 255, 255), -1)
    cv2.rectangle(image, (left, top), (left + panel_w, top + panel_h), (0, 0, 0), 1)

    for i, (name, color) in enumerate(entries):
        y = top + 18 + i * line_h
        cv2.rectangle(image, (left + 10, y - 8), (left + 26, y + 8), color, -1)
        cv2.putText(
            image,
            name,
            (left + 34, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def vision_check():
    os.makedirs(tmp_dir, exist_ok=True)
    if not os.path.isdir(prompt_dir):
        return

    json_files = sorted([f for f in os.listdir(prompt_dir) if f.lower().endswith(".json")])
    for json_name in json_files:
        json_path = os.path.join(prompt_dir, json_name)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_name = data.get("file_name")
        if not file_name:
            continue

        file_path = os.path.join(mask_dir, file_name)
        if not os.path.isfile(file_path):
            continue

        mask = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue

        mask01 = to_binary_mask_512(mask)
        canvas = (mask01 * 255).astype(np.uint8)
        vis = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

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

        out_name = os.path.splitext(json_name)[0] + "_vis.png"
        out_path = os.path.join(tmp_dir, out_name)
        cv2.imwrite(out_path, vis)


def main():
    os.makedirs(prompt_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    file_list = sorted(os.listdir(mask_dir))

    for file_name in file_list:
        file_path = os.path.join(mask_dir, file_name)
        if not os.path.isfile(file_path):
            continue

        mask = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue

        mask01 = to_binary_mask_512(mask)

        largest = get_largest_component(mask01)
        if largest is None:
            continue

        ys, xs = np.where(largest == 1)
        if xs.size == 0:
            continue

        xs_i = xs.astype(np.int32)
        ys_i = ys.astype(np.int32)
        xs_f = xs.astype(np.float32)
        ys_f = ys.astype(np.float32)

        # 3) box from the largest connected component
        x_min = float(xs_f.min())
        x_max = float(xs_f.max())
        y_min = float(ys_f.min())
        y_max = float(ys_f.max())
        box = [round(x_min, 4), round(y_min, 4), round(x_max, 4), round(y_max, 4)]

        box_w = x_max - x_min + 1.0
        box_h = y_max - y_min + 1.0
        box_cx = (x_min + x_max) / 2.0
        box_cy = (y_min + y_max) / 2.0

        # 1) centroid of largest component; snap to nearest fg pixel to ensure point is on mask==1.
        centroid_fx = float(xs_f.mean())
        centroid_fy = float(ys_f.mean())
        cx, cy = nearest_fg_point(centroid_fx, centroid_fy, xs_f, ys_f)
        centroid = [int(cx), int(cy)]

        # 2) centroid with <=30% offset; choose from fg candidates to ensure on mask==1.
        cands_x, cands_y = get_fg_candidates_within_offset(cx, cy, xs_i, ys_i, box_w, box_h)
        rand_idx = int(rng.integers(0, cands_x.size))
        centroid_with_offset = [int(cands_x[rand_idx]), int(cands_y[rand_idx])]

        # 4) box with <=30% width/height size perturbation (center fixed).
        dw = float(rng.uniform(-0.3 * box_w, 0.3 * box_w))
        dh = float(rng.uniform(-0.3 * box_h, 0.3 * box_h))
        box_with_size_offset = box_from_center_size(box_cx, box_cy, box_w + dw, box_h + dh)

        # 5) box with <=30% horizontal/vertical position perturbation (size fixed).
        shift_x = float(rng.uniform(-0.3 * box_w, 0.3 * box_w))
        shift_y = float(rng.uniform(-0.3 * box_h, 0.3 * box_h))
        box_with_position_offset = box_from_center_size(box_cx + shift_x, box_cy + shift_y, box_w, box_h)

        data = {
            "file_name": file_name,
            "image_size": [512, 512],
            "seed": seed,
            "centroid": centroid,
            "centroid_with_offset": centroid_with_offset,
            "box": box,
            "box_with_size_offset": box_with_size_offset,
            "box_with_position_offset": box_with_position_offset,
        }

        out_name = os.path.splitext(file_name)[0] + ".json"
        out_path = os.path.join(prompt_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # main()
    vision_check()