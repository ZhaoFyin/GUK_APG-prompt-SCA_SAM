import os
import cv2
from draw_utils import overlap
from utils import read_txt_to_dict, get_latest_epoch_pth
from torchvision.transforms import transforms, InterpolationMode
from dataset import CustomDataset
from torch.utils.data import DataLoader
import torch
from MakeModel import KgeScaSam
import matplotlib.pyplot as plt
import json
import matplotlib.patches as patches


ids = [1, 12, 13]

data_dir = r"G:\KGE-SwinFpn/VOCdevkit_YYL"
mtds = ["auto", "centroid", "centroid_with_offset", "box", "box_with_size_offset", "box_with_position_offset"]
result_dirs = {"auto": "./vis_results/auto",
               "centroid": "./vis_results/centroid",
               "centroid_with_offset": "./vis_results/centroid_with_offset",
               "box": "./vis_results/box",
               "box_with_size_offset": "./vis_results/box_with_size_offset",
               "box_with_position_offset": "./vis_results/box_with_position_offset"}
auto_dir = "./results/YYL/auto"
pmt_dir = "./prompt"


with open(os.path.join(data_dir, r"VOC_landslide\ImageSets\Segmentation\test.txt"), "r") as f:
    file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

draw_names = [file_names[idx] for idx in ids]
subplot_data = {}
rgb_dir = os.path.join(data_dir, "VOC_landslide", 'JPEGImages')
mask_dir = os.path.join(data_dir, "VOC_landslide", 'SegmentationObject')
transform = transforms.Compose([
    transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

for draw_name in draw_names:
    img_name = os.path.join(rgb_dir, draw_name + ".jpg")
    mask_name = os.path.join(mask_dir, draw_name + ".png")

    img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    mask[mask != 0] = 1
    subplot_data[draw_name] = [overlap(img, mask, mask)]

    with open(os.path.join(data_dir, r"VOC_landslide\ImageSets\Segmentation\tmp.txt"), 'w', encoding='utf-8') as f:
        f.write(draw_name + '\n')

    dataset = CustomDataset(voc_root=data_dir, kge=True, txt_name='tmp.txt', transform=transform, oh=True)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    cfg = read_txt_to_dict(os.path.join(auto_dir, "config.txt"))
    net = KgeScaSam(eval(cfg["lora_cfg"]), num_classes=eval(cfg["num_classes"]), sca=eval(cfg["sca"]),
                    threshold=eval(cfg["thr"]), pmt_mode=cfg["prompt_mode"], pmt_set=cfg["prompt_set"],
                    num_k=9, img_s=512)
    weight = torch.load(get_latest_epoch_pth(auto_dir))
    net.load_state_dict(weight, strict=True)

    net.cuda()
    with torch.no_grad():

        for sampled_batch in loader:
            image_batch, label_batch, k_batch, s_batch, p_batch, filename = sampled_batch

            image_batch, label_batch, k_batch, s_batch = image_batch.cuda(), label_batch.cuda(), k_batch.cuda(), s_batch.cuda()
            outputs, prompts = net(image_batch, k_batch, s_batch, p_batch)

            logits = prompts["mask"][0].detach().cpu().numpy()
            subplot_data[draw_name].append(logits)
    os.remove(os.path.join(data_dir, r"VOC_landslide\ImageSets\Segmentation\tmp.txt"))
    for mtd in mtds:
        pred_name = os.path.join(result_dirs[mtd], draw_name + ".png")
        pred_img = cv2.imread(pred_name, cv2.IMREAD_UNCHANGED)
        subplot_data[draw_name].append(overlap(img, pred_img, mask))


win_size = 800
h_gap = 60
w_gap = 60
dpi = 500
h_c = len(ids)
w_c = 8
w_px = w_c * (win_size + w_gap) + 2 * w_gap
h_px = h_c * (win_size + h_gap) + 2 * h_gap

fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
x_list = ["（a）真实滑坡掩膜",
          "（b）自动提示词提示",
          "（c）本章结果",
          "（d）标准质心",
          "（e）偏移质心",
          "（f）标准外接框",
          "（g）尺寸偏移外接框",
          "（h）位置偏移外接框"]

for h in range(h_c):
    pmt_path = os.path.join(pmt_dir, draw_names[h] + ".json")
    with open(pmt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for w in range(w_c):
        ax = fig.add_axes([(2 * w_gap + w * (w_gap + win_size)) / w_px,
                           (h_px - (h + 1) * (h_gap + win_size)) / h_px,
                           win_size / w_px, win_size / h_px])
        ax.imshow(subplot_data[draw_names[h]][w])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_visible(False)
        if h == h_c-1:
            ax.set_xlabel(x_list[w], fontsize=12, fontname='SimSun')
        if w == 0:
            ax.set_ylabel(f"滑坡示例 {h+1}", fontsize=12, fontname='SimSun')

        if mtds[w-2] in ["centroid", "centroid_with_offset"] and w >= 2:
            point = data.get(mtds[w-2], None)
            ax.scatter(point[0], point[1], c='yellow', s=15)

        elif mtds[w-2] in ["box", "box_with_size_offset", "box_with_position_offset"] and w >= 2:
            box = data.get(mtds[w-2], None)
            x1, y1, x2, y2 = box

            rect = patches.Rectangle(
                (x1, y1),           # 左上角
                x2 - x1,            # 宽
                y2 - y1,            # 高
                linewidth=1,
                edgecolor='yellow',
                facecolor='none')
            ax.add_patch(rect)


plt.savefig(f'fig/Diff_pmt_result.png')
plt.close()
