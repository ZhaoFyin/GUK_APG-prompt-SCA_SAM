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
from draw_utils import diff_color


data_dir = r"G:\KGE-SwinFpn/VOCdevkit_YYL"
auto_dir = "./results/YYL/auto"

cfg = read_txt_to_dict(os.path.join(auto_dir, "config.txt"))
net = KgeScaSam(eval(cfg["lora_cfg"]), num_classes=eval(cfg["num_classes"]), sca=eval(cfg["sca"]),
                threshold=eval(cfg["thr"]), pmt_mode=cfg["prompt_mode"], pmt_set=cfg["prompt_set"],
                num_k=9, img_s=512, post_mode="unit")

weight = torch.load(get_latest_epoch_pth(auto_dir))
net.load_state_dict(weight, strict=True)
net.cuda()

def main(ids):
    with open(os.path.join(data_dir, r"VOC_landslide\ImageSets\Segmentation\train.txt"), "r") as f:
        file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

    draw_name = file_names[ids]

    subplot_data = []
    rgb_dir = os.path.join(data_dir, "VOC_landslide", 'JPEGImages')
    mask_dir = os.path.join(data_dir, "VOC_landslide", 'SegmentationObject')

    with open(os.path.join(data_dir, r"VOC_landslide\ImageSets\Segmentation\tmp.txt"), 'w', encoding='utf-8') as f:
        f.write(draw_name + '\n')
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    dataset = CustomDataset(voc_root=data_dir, kge=True, txt_name='tmp.txt', transform=transform, oh=True, sp="_RGB")
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    img_name = os.path.join(rgb_dir, draw_name + ".jpg")
    mask_name = os.path.join(mask_dir, draw_name + ".png")

    img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    mask[mask != 0] = 1
    ol = overlap(img, mask, mask)
    subplot_data.append(ol)

    with torch.no_grad():
        for sampled_batch in loader:
            image_batch, _, k_batch, s_batch, p_batch, filename = sampled_batch

            image_batch, k_batch, s_batch = image_batch.cuda(), k_batch.cuda(), s_batch.cuda()
            outputs, prompts = net(image_batch, k_batch, s_batch, p_batch)

            out_logits = torch.sigmoid(outputs['masks'])
            pre_mask = (out_logits > 0.5) * 1

            pred = pre_mask[0][0].detach().cpu().numpy()

            subplot_data.append(diff_color(s_batch[0][0].detach().cpu().numpy()))
            subplot_data.append(prompts["mask"][0].detach().cpu().numpy())
            subplot_data.append(overlap(img, pred, mask))

    os.remove(os.path.join(data_dir, r"VOC_landslide\ImageSets\Segmentation\tmp.txt"))

    win_size = 800
    h_gap = 75
    w_gap = 75
    dpi = 500
    h_c = 1
    w_c = 4
    w_px = w_c * (win_size + w_gap) + w_gap
    h_px = h_c * (win_size + h_gap) + 2 * h_gap

    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)

    x_name = ["(a) 影像与标注叠加",
              "(b) 地理单元划分",
              "(c) 提示",
              "(d) 滑坡分割结果",]

    for w in range(w_c):

        ax = fig.add_axes([(1 * w_gap + w * (w_gap + win_size)) / w_px,
                           (h_px - win_size - w_gap) / h_px,
                           win_size / w_px, win_size / h_px])
        ax.imshow(subplot_data[w])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(x_name[w], fontsize=12, fontname='SimSun')
    plt.savefig(f'fig/Error_cls.png')
    plt.close()

if __name__ == "__main__":

    main(319)