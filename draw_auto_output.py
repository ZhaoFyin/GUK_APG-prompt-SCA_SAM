import os
from torchvision.transforms import transforms, InterpolationMode
from dataset import CustomDataset
from torch.utils.data import DataLoader
import torch
from MakeModel import KgeScaSam
import torch.nn.functional as F
from draw_utils import overlap
import cv2
import matplotlib.pyplot as plt


ids = [0, 1]
data_dir = r"G:\KGE-SwinFpn/VOCdevkit_YYL"
gt_data = {}

rgb_dir = os.path.join(data_dir, "VOC_landslide", 'JPEGImages')
mask_dir = os.path.join(data_dir, "VOC_landslide", 'SegmentationObject')
dem_dir = os.path.join(data_dir, "VOC_landslide", 'Knowledge', 'dem_img')
split50_dir = os.path.join(data_dir, "VOC_landslide", 'Split50')

with open(os.path.join(data_dir, r"VOC_landslide\ImageSets\Segmentation\test.txt"), "r") as f:
    file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

draw_names = [file_names[idx] for idx in ids]
for draw_name in draw_names:
    img_name = os.path.join(rgb_dir, draw_name + ".jpg")
    mask_name = os.path.join(mask_dir, draw_name + ".png")

    img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    mask[mask != 0] = 1
    gt_data[draw_name] = overlap(img, mask, mask)

with open(os.path.join(data_dir, r"VOC_landslide\ImageSets\Segmentation\tmp.txt"), 'w', encoding='utf-8') as f:
    for x in draw_names:
        f.write(x + '\n')

transform = transforms.Compose([
    transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
])


from utils import read_txt_to_dict, get_latest_epoch_pth
cfg = read_txt_to_dict("results/YYL/auto/config.txt")

net = KgeScaSam(eval(cfg["lora_cfg"]), num_classes=eval(cfg["num_classes"]), sca=eval(cfg["sca"]),
                threshold=eval(cfg["thr"]), pmt_mode=cfg["prompt_mode"], pmt_set=cfg["prompt_set"],
                num_k=9, img_s=512, post_mode="unit")
weight = torch.load(get_latest_epoch_pth("results/YYL/auto"))
net.load_state_dict(weight, strict=True)
net.cuda()


auto_results = {}
dataset = CustomDataset(voc_root=data_dir, kge=True, txt_name='tmp.txt', transform=transform, oh=True, sp="50")
loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
with torch.no_grad():

    for sampled_batch in loader:
        image_batch, label_batch, k_batch, s_batch, p_batch, filename = sampled_batch

        image_batch, label_batch, k_batch, s_batch = image_batch.cuda(), label_batch.cuda(), k_batch.cuda(), s_batch.cuda()
        outputs, prompts = net(image_batch, k_batch, s_batch, p_batch)
        for i in range(image_batch.shape[0]):
            logits = prompts["mask"][i]
            auto_results[filename[i]] = [logits.detach().cpu().numpy()]

auto_results100 = {}
dataset = CustomDataset(voc_root=data_dir, kge=True, txt_name='tmp.txt', transform=transform, oh=True, sp="100")
loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
with torch.no_grad():

    for sampled_batch in loader:
        image_batch, label_batch, k_batch, s_batch, p_batch, filename = sampled_batch

        image_batch, label_batch, k_batch, s_batch = image_batch.cuda(), label_batch.cuda(), k_batch.cuda(), s_batch.cuda()
        outputs, prompts = net(image_batch, k_batch, s_batch, p_batch)
        for i in range(image_batch.shape[0]):
            logits = prompts["mask"][i]
            auto_results[filename[i]].append(logits.detach().cpu().numpy())

dataset = CustomDataset(voc_root=data_dir, kge=True, txt_name='tmp.txt', transform=transform, oh=True, sp="200")
loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
with torch.no_grad():

    for sampled_batch in loader:
        image_batch, label_batch, k_batch, s_batch, p_batch, filename = sampled_batch

        image_batch, label_batch, k_batch, s_batch = image_batch.cuda(), label_batch.cuda(), k_batch.cuda(), s_batch.cuda()
        outputs, prompts = net(image_batch, k_batch, s_batch, p_batch)
        for i in range(image_batch.shape[0]):
            logits = prompts["mask"][i]
            auto_results[filename[i]].append(logits.detach().cpu().numpy())

win_size = 800
h_gap = 60
w_gap = 60
dpi = 500
h_c = 2
w_c = 4
w_px = w_c * (win_size + w_gap) + 2 * w_gap
h_px = h_c * (win_size + h_gap) + 3 * h_gap


fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
y_list = ["滑坡示例 1", "滑坡示例 2"]
x_list = ["（a）RGB影像与滑坡掩膜叠加图",
          "（b）自动提示词生成器提示\n  50单元",
          "（c）自动提示词生成器提示\n  100单元",
          "（d）自动提示词生成器提示\n  200单元"]

for h in range(h_c):
    for w in range(w_c):
        ax = fig.add_axes([(2 * w_gap + w * (w_gap + win_size)) / w_px,
                           (h_px - (h + 1) * (h_gap + win_size)) / h_px,
                           win_size / w_px, win_size / h_px])
        if w == 0:
            subplot_data = gt_data[draw_names[h]]
            ax.imshow(subplot_data)
        else:
            subplot_data = auto_results[draw_names[h]][w-1]
            im = ax.imshow(subplot_data, cmap='viridis')
            im_for_cbar = im

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_visible(False)
        if h == 1:
            ax.set_xlabel(x_list[w], fontsize=8, fontname='SimSun')
        if w == 0:
            ax.set_ylabel(y_list[h], fontsize=8, fontname='SimSun')

plt.savefig(f'fig/Auto_result.png')
plt.close()
os.remove(os.path.join(data_dir, r"VOC_landslide\ImageSets\Segmentation\tmp.txt"))