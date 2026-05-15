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


ids = 1

data_dir = r"G:\KGE-SwinFpn/VOCdevkit_YYL"
auto_dir = "./results/YYL/auto"


with open(os.path.join(data_dir, r"VOC_landslide\ImageSets\Segmentation\test.txt"), "r") as f:
    file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

draw_name = file_names[ids]

subplot_data = [
    [None, None, None],
    [None, None, None],
    [None, None, None]
]
rgb_dir = os.path.join(data_dir, "VOC_landslide", 'JPEGImages')
mask_dir = os.path.join(data_dir, "VOC_landslide", 'SegmentationObject')
transform = transforms.Compose([
    transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
])
with open(os.path.join(data_dir, r"VOC_landslide\ImageSets\Segmentation\tmp.txt"), 'w', encoding='utf-8') as f:
    f.write(draw_name + '\n')


img_name = os.path.join(rgb_dir, draw_name + ".jpg")
mask_name = os.path.join(mask_dir, draw_name + ".png")


img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
mask[mask != 0] = 1
ol = overlap(img, mask, mask)
subplot_data[0][0] = img
subplot_data[1][0] = mask
subplot_data[2][0] = ol

cfg = read_txt_to_dict(os.path.join(auto_dir, "config.txt"))
unit_net = KgeScaSam(eval(cfg["lora_cfg"]), num_classes=eval(cfg["num_classes"]), sca=eval(cfg["sca"]),
                     threshold=eval(cfg["thr"]), pmt_mode=cfg["prompt_mode"], pmt_set=cfg["prompt_set"],
                     num_k=9, img_s=512, post_mode="unit")
window_net = KgeScaSam(eval(cfg["lora_cfg"]), num_classes=eval(cfg["num_classes"]), sca=eval(cfg["sca"]),
                       threshold=eval(cfg["thr"]), pmt_mode=cfg["prompt_mode"], pmt_set=cfg["prompt_set"],
                       num_k=9, img_s=512, post_mode="window")
none_net = KgeScaSam(eval(cfg["lora_cfg"]), num_classes=eval(cfg["num_classes"]), sca=eval(cfg["sca"]),
                     threshold=eval(cfg["thr"]), pmt_mode=cfg["prompt_mode"], pmt_set=cfg["prompt_set"],
                     num_k=9, img_s=512, post_mode=None)


dataset = CustomDataset(voc_root=data_dir, kge=True, txt_name='tmp.txt', transform=transform, oh=True)
loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

weight = torch.load(get_latest_epoch_pth(auto_dir))
unit_net.load_state_dict(weight, strict=True)
unit_net.cuda()

window_net.load_state_dict(weight, strict=True)
window_net.cuda()

none_net.load_state_dict(weight, strict=False)
none_net.cuda()


with torch.no_grad():

    for sampled_batch in loader:
        image_batch, _, k_batch, s_batch, p_batch, filename = sampled_batch

        image_batch, k_batch, s_batch = image_batch.cuda(), k_batch.cuda(), s_batch.cuda()
        unit_outputs, unit_prompts = unit_net(image_batch, k_batch, s_batch, p_batch)
        window_outputs, window_prompts = window_net(image_batch, k_batch, s_batch, p_batch)
        none_outputs, none_prompts = none_net(image_batch, k_batch, s_batch, p_batch)


        unit_mask = ((torch.sigmoid(unit_outputs['masks']) > 0.5) * 1)[0][0].detach().cpu().numpy()
        window_mask = ((torch.sigmoid(window_outputs['masks']) > 0.5) * 1)[0][0].detach().cpu().numpy()
        none_mask = ((torch.sigmoid(none_outputs['masks']) > 0.5) * 1)[0][0].detach().cpu().numpy()

        subplot_data[0][1] = unit_prompts['mask'].squeeze().detach().cpu().numpy()
        subplot_data[1][1] = window_prompts['mask'].squeeze().detach().cpu().numpy()
        subplot_data[2][1] = none_prompts['mask'].squeeze().detach().cpu().numpy()

        subplot_data[0][2] = overlap(img, unit_mask, mask)
        subplot_data[1][2] = overlap(img, window_mask, mask)
        subplot_data[2][2] = overlap(img, none_mask, mask)
os.remove(os.path.join(data_dir, r"VOC_landslide\ImageSets\Segmentation\tmp.txt"))

win_size = 800
h_gap = 260
w_gap = 75
dpi = 500
h_c = 3
w_c = 3
w_px = w_c * (win_size + w_gap) + 2 * w_gap
h_px = h_c * (win_size + h_gap) + w_gap

fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)

x_name = ["(a) 遥感影像",
          "(b) 滑坡标注掩膜",
          "(c) 影像与标注叠加",
          "提示",
          "提示",
          "提示",
          "分割结果",
          "分割结果",
          "分割结果",]

group_name = [
    "（d）地理单元注意力",
    "（e）标准窗口注意力",
    "（f）无"
]
axes_list = []

for h in range(h_c):
    row_axes = []
    for w in range(w_c):
        shift = 0 if w == 0 else w_gap
        ax = fig.add_axes([(1 * w_gap + w * (w_gap + win_size) + shift) / w_px,
                           (h_px - (h + 1) * (h_gap + win_size) + h_gap - w_gap) / h_px,
                           win_size / w_px, win_size / h_px])
        ax.imshow(subplot_data[h][w])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xlabel(x_name[h+3*w], fontsize=12 if w == 0 else 9, fontname='SimSun')
        row_axes.append(ax)

    axes_list.append(row_axes)


from matplotlib.lines import Line2D

# =========================
# 在每一行右侧两图中间下方添加说明文字
# =========================
for h in range(h_c):
    ax_mid = axes_list[h][1]   # 第2列
    ax_right = axes_list[h][2] # 第3列

    pos_mid = ax_mid.get_position()
    pos_right = ax_right.get_position()

    # 两个图整体的水平中心
    x_center = (pos_mid.x0 + pos_right.x1) / 2

    # 放在“提示/分割结果”这两个xlabel的下方
    y_text = min(pos_mid.y0, pos_right.y0) - 100 / h_px

    fig.text(
        x_center, y_text, group_name[h],
        ha='center', va='top',
        fontsize=12, fontname='SimSun'
    )

# =========================
# 1. 第一列与其余列之间的竖直分割线
# =========================
x_div = (1 * w_gap + win_size + w_gap) / w_px

y_top = 1
y_bottom = w_gap/h_px

fig.add_artist(Line2D(
    [x_div, x_div], [y_bottom, y_top],
    transform=fig.transFigure,
    color='black',
    linewidth=1.5,
    linestyle='--'
))

# =========================
# 2. 除第一列外，每一行之间的水平分割线
#    只画在第2列到最后1列的区域
# =========================
x_left = (1 * w_gap + win_size + w_gap) / w_px
x_right = 1

for h in range(h_c - 1):
    # 第 h 行和第 h+1 行之间的分界线
    y_div = (h_px - w_gap - (h + 1) * (h_gap + win_size) + 50) / h_px


    fig.add_artist(Line2D(
        [x_left, x_right], [y_div, y_div],
        transform=fig.transFigure,
        color='black',
        linewidth=1.5,
        linestyle='--'
    ))

plt.savefig(f'fig/Diff_postprocess_result.png')
plt.close()