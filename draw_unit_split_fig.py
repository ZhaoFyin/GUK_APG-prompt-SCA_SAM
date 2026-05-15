import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from draw_utils import *


voc_dir = r"G:\KGE-SwinFpn/VOCdevkit_YYL"
pmt_dir="./prompt"
idx = 19

rgb_dir = os.path.join(voc_dir, "VOC_landslide", 'JPEGImages')
mask_dir = os.path.join(voc_dir, "VOC_landslide", 'SegmentationObject')
dem_dir = os.path.join(voc_dir, "VOC_landslide", 'Knowledge', 'dem_img')
split50_dir = os.path.join(voc_dir, "VOC_landslide", 'Split50')
split100_dir = os.path.join(voc_dir, "VOC_landslide", 'Split100')
split200_dir = os.path.join(voc_dir, "VOC_landslide", 'Split200')


file_name = os.listdir(rgb_dir)[idx].split(".")[0]

img_name = os.path.join(rgb_dir, file_name + ".jpg")
mask_name = os.path.join(mask_dir, file_name + ".png")
split50_name = os.path.join(split50_dir, file_name + ".png")
split100_name = os.path.join(split100_dir, file_name + ".png")
split200_name = os.path.join(split200_dir, file_name + ".png")
dem_name = os.path.join(dem_dir, file_name + ".jpg")

img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
mask[mask!=0] = 1

split50 = cv2.imread(split50_name, cv2.IMREAD_UNCHANGED)
split50 = cv2.resize(split50, (512, 512), interpolation=cv2.INTER_NEAREST)
split100 = cv2.imread(split100_name, cv2.IMREAD_UNCHANGED)
split100 = cv2.resize(split100, (512, 512), interpolation=cv2.INTER_NEAREST)
split200 = cv2.imread(split200_name, cv2.IMREAD_UNCHANGED)
split200 = cv2.resize(split200, (512, 512), interpolation=cv2.INTER_NEAREST)

dem = cv2.imread(dem_name, cv2.IMREAD_UNCHANGED)
dem = cv2.resize(dem, (512, 512), interpolation=cv2.INTER_LINEAR)


subplot_data = [overlap(img, mask, mask), dem, diff_color(split50), diff_color(split100), diff_color(split200)]

win_size = 800
h_gap = 60
w_gap = 60
dpi = 500
h_c = 1
w_c = 5
w_px = w_c * (win_size + w_gap) + w_gap
h_px = h_c * (win_size + 2 * h_gap) + 2 * h_gap

fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)

ax = fig.add_axes([(w_gap + 0 * (w_gap + win_size))/ w_px, (h_px - w_gap - win_size) / h_px, win_size / w_px, win_size / h_px])
ax.imshow(subplot_data[0])
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xlabel("（a）RGB影像与滑坡掩膜叠加图", fontsize=8, fontname='SimSun')

ax = fig.add_axes([(w_gap + 1 * (w_gap + win_size))/ w_px, (h_px - w_gap - win_size) / h_px, win_size / w_px, win_size / h_px])
ax.imshow(subplot_data[1], cmap='gray')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xlabel("（b）数字高程模型", fontsize=8, fontname='SimSun')

ax = fig.add_axes([(w_gap + 2 * (w_gap + win_size))/ w_px, (h_px - w_gap - win_size) / h_px, win_size / w_px, win_size / h_px])
ax.imshow(subplot_data[2])
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xlabel("（c）地理单元划分\n  50单元", fontsize=8, fontname='SimSun')

ax = fig.add_axes([(w_gap + 3 * (w_gap + win_size))/ w_px, (h_px - w_gap - win_size) / h_px, win_size / w_px, win_size / h_px])
ax.imshow(subplot_data[3])
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xlabel("（d）地理单元划分\n  100单元", fontsize=8, fontname='SimSun')

ax = fig.add_axes([(w_gap + 4 * (w_gap + win_size))/ w_px, (h_px - w_gap - win_size) / h_px, win_size / w_px, win_size / h_px])
ax.imshow(subplot_data[4])
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xlabel("（e）地理单元划分\n  200单元", fontsize=8, fontname='SimSun')


plt.savefig(f'fig/Unit_split.png')
plt.close()
