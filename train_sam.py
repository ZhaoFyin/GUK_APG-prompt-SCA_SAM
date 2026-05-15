import warnings
warnings.filterwarnings("ignore")

import argparse
from dataset import CustomDataset
import random
import datetime
import torch
from torchvision.transforms import transforms, InterpolationMode
import os
from train_utils import trainer_synapse
from utils import Tee
from torch.utils.data import DataLoader
from MakeModel import KgeScaSam

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True


lora_cfg = {
    "enable": True,
    "sca_enable": True,
    "r": 8, "alpha": 16, "dropout": 0.0,
    "target_modules": ["qkv", "proj"],
    "target_blocks": "indices",
    "indices": [8, 9, 10, 11],
    "lr_rate": 1
}

dataset_name = "YYL"
data_dict = {"YYL": r"G:\KGE-SwinFpn/VOCdevkit_YYL"}


def main(args, snapshot_path):
    CLASSES = 1

    net = KgeScaSam(lora_cfg, num_classes=args.num_classes, sca=args.sca, threshold=args.thr,
                    pmt_mode=args.prompt_mode, pmt_set=args.prompt_set,
                    num_k=9, img_s=512)
    net.cuda()

    print(f"vit可训练参数数量: {sum(p.numel() for p in net.sam.image_encoder.parameters() if p.requires_grad):,}")
    print(f"总可训练参数数量: {sum(p.numel() for p in net.sam.parameters() if p.requires_grad):,}")
    print(f"总参数数量: {sum(p.numel() for p in net.sam.image_encoder.parameters()):,}")

    multimask_output = False

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset(voc_root=args.data_path, kge=True, txt_name="train.txt",
                                  transform=transform)
    val_dataset = CustomDataset(voc_root=args.data_path, kge=True, txt_name="val.txt",
                                transform=transform)


    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=6,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    trainer_synapse(args, net, snapshot_path, train_loader, val_loader, multimask_output, CLASSES)


def parameter(mode, setting):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='./results')
    parser.add_argument('--dataset', type=str,
                        default=dataset_name, help='dataset_name')
    parser.add_argument('--data_path', type=str,
                        default=data_dict[dataset_name], help='dataset_name')

    parser.add_argument('--experiment', type=str,
                        default='MySAM', help='experiment_name')

    parser.add_argument('--num_classes', type=int,
                        default=1, help='output channel of network')
    parser.add_argument('--max_epochs', type=int,
                        default=200, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=2, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=0.0006,
                        help='segmentation network learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay')
    parser.add_argument('--img_size', type=int,
                        default=512, help='input patch size of network input')
    parser.add_argument('--input_size', type=int, default=1024, help='The input size for training SAM')
    parser.add_argument('--warmup', action='store_true', default=True,
                        help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--warmup_period', type=int, default=100,
                        help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('--AdamW', action='store_true', default=True,
                        help='If activated, use AdamW')
    parser.add_argument('--sca', default=True)
    parser.add_argument('--prompt_mode', default=mode, choices=["mask", "point", "box"])

    parser.add_argument('--prompt_set', default=setting,
                        choices=["auto",
                                 "centroid",  # 标准质心
                                 "centroid_with_offset",  # 偏移质心
                                 "box",  # 标准框
                                 "box_with_size_offset",  # 尺度偏移框
                                 "box_with_position_offset",  # 位置偏移框
                                 ])
    parser.add_argument('--thr', default=None)
    parser.add_argument('--lora_cfg', default=lora_cfg)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    you_want_train = "本文模型"
    prompt_setting = {
        "本文模型": ["mask", "auto"],
        "标准质心": ["point", "centroid"],
        "偏移质心": ["point", "centroid_with_offset"],
        "标准框": ["box", "box"],
        "尺度偏移框": ["box", "box_with_size_offset"],
        "位置偏移框": ["box", "box_with_position_offset"],
    }
    mod, setting = prompt_setting[you_want_train]

    args = parameter(mod, setting)
    now = datetime.datetime.now()
    now = now.strftime("%m%d_%H%M%S")
    snapshot_path = os.path.join(args.output, "{}".format(args.dataset), now)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    with Tee(os.path.join(snapshot_path, "running.txt"), 'w'):
        main(args, snapshot_path)