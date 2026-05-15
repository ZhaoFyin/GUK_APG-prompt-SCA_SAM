import warnings
warnings.filterwarnings("ignore")

from utils import Accuracy, read_txt_to_dict, get_latest_epoch_pth
import os
from dataset import CustomDataset
import numpy as np
import PIL.Image as Image
import torch
from torchvision.transforms import transforms, InterpolationMode
from torch.utils.data import DataLoader
from MakeModel import KgeScaSam
from tqdm import tqdm


dataset_name = "YYL"
sp_count = "200"
post_mode = "unit"
data_dict = {"YYL": r"G:\KGE-SwinFpn/VOCdevkit_YYL"}

def eval_element(eval_dir):
    cfg = read_txt_to_dict(os.path.join(eval_dir, "config.txt"))

    net = KgeScaSam(eval(cfg["lora_cfg"]), num_classes=eval(cfg["num_classes"]), sca=eval(cfg["sca"]),
                    threshold=eval(cfg["thr"]), pmt_mode=cfg["prompt_mode"], pmt_set=cfg["prompt_set"],
                    num_k=9, img_s=512, post_mode=post_mode)


    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    test_dataset = CustomDataset(voc_root=data_dict[dataset_name], kge=True, txt_name="test.txt",
                                transform=transform, sp=sp_count)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    weight = torch.load(get_latest_epoch_pth(eval_dir))

    net.load_state_dict(weight, strict=False)
    net.cuda()
    accuracy = Accuracy()

    t = tqdm(test_loader, desc=f"testing...", leave=False, dynamic_ncols=True)
    save_dir = os.path.join("vis_results", cfg["prompt_set"]+sp_count)
    # save_dir = os.path.join("vis_results", "ProcessNone")
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for sampled_batch in t:
            image_batch, label_batch, k_batch, s_batch, p_batch, filename = sampled_batch

            image_batch, label_batch, k_batch, s_batch = image_batch.cuda(), label_batch.cuda(), k_batch.cuda(), s_batch.cuda()
            outputs, prompts = net(image_batch, k_batch, s_batch, p_batch)
            out_logits = torch.sigmoid(outputs['masks'])
            pre_mask = (out_logits > 0.5) * 1
            accuracy.update(pre_mask, label_batch)
            for i in range(label_batch.shape[0]):
                p = pre_mask[i].squeeze(0).cpu().numpy().astype(np.uint8)
                img = Image.fromarray(p)
                img.save(os.path.join(save_dir, f"{filename[i]}.png"))

        pa, mpa, miou = accuracy.calculate_metrics()
        print("pa: {}, mpa: {}, miou: {}".format(pa, mpa, miou))

    print(f"{save_dir} saved !")

if __name__ == "__main__":
    # ls = ["auto",
    #       "centroid",  # 质心
    #       "centroid_with_offset",  # 偏移质心
    #       "box",  # 框
    #       "box_with_size_offset",  # 尺度偏移框
    #       "box_with_position_offset",  # 位置偏移框
    # ]
    ls = ["auto"]
    for l in ls:
        eval_element(f"./results/YYL/{l}")


"""
pa: 0.9867490485862449, mpa: 0.9419796096091304, miou: 0.9167307884302418
pa: 0.9812393894901982, mpa: 0.9312201909780501, miou: 0.8869629644298134
pa: 0.9804535971747504, mpa: 0.9233315243252287, miou: 0.8815603740501095
pa: 0.9827129575941298, mpa: 0.9325909548183469, miou: 0.8943931587235425
pa: 0.9817433887057834, mpa: 0.9293206521883040, miou: 0.8890035168609780
pa: 0.9812390362774884, mpa: 0.9276801469913295, miou: 0.8862326736855759
"""

"""
50:
pa: 0.9821016876785843, mpa: 0.9175153371907396, miou: 0.8883195619351065
100:
pa: 0.9867490485862449, mpa: 0.9419796096091304, miou: 0.9167307884302418 
200
pa: 0.9854381349351671, mpa: 0.9378588936596677, miou: 0.9091548376330940
"""


"""
unit:
pa: 0.9867490485862449, mpa: 0.9419796096091304, miou: 0.9167307884302418 
window:
pa: 0.9847668188589590, mpa: 0.9308316790900899, miou: 0.9044342159620202
None
pa: 0.9743398030598959, mpa: 0.8912740295443401, miou: 0.8459338972455959
"""
