import os.path
import os
import numpy as np
import sys
from tqdm import tqdm
import re
import torch
import torch.nn.functional as F
from PIL import Image


# 精度指标
class Accuracy:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self._cm = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    def update(self, predictions, labels):
        pred = predictions.reshape(-1).to(torch.int64).cpu()
        lab = labels.reshape(-1).to(torch.int64).cpu()
        valid = (lab >= 0) & (lab < self.num_classes)
        if not torch.any(valid):
            return
        pred = pred[valid]
        lab = lab[valid]
        idx = lab * self.num_classes + pred
        cm = torch.bincount(idx, minlength=self.num_classes ** 2)
        self._cm += cm.reshape(self.num_classes, self.num_classes)

    def calculate_metrics(self, predictions=None, labels=None):
        if predictions is not None and labels is not None:
            self.reset()
            self.update(predictions, labels)
        return self._from_cm()

    def _from_cm(self):
        cm = self._cm.to(torch.float64)
        total = cm.sum()
        if total == 0:
            return 0.0, 0.0, 0.0
        correct = torch.diag(cm).sum()
        pa = (correct / total).item()
        class_correct = torch.diag(cm) / (cm.sum(dim=1) + 1e-10)
        mpa = torch.nanmean(class_correct).item()
        union = (cm.sum(dim=1) + cm.sum(dim=0) - torch.diag(cm))
        iou = torch.diag(cm) / (union + 1e-10)
        miou = torch.nanmean(iou).item()
        return pa, mpa, miou


class Tee:
    def __init__(self, filename, mode='w'):
        self.filename = filename
        self.mode = mode
        self.stdout = sys.stdout
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout
        if self.file:
            self.file.close()

    def write(self, data):
        if self.file:
            self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        if self.file:
            self.file.flush()


def read_txt_to_dict(file_path, encoding="ANSI"):
    config_dict = {}
    with open(file_path, 'r', encoding=encoding) as file:
        for line in file:
            key, value = line.strip().split(':', 1)  # 按照冒号分割，限制为一次分割
            config_dict[key.strip()] = value.strip()  # 去除两端的空格并保存到字典中
    return config_dict


def get_latest_epoch_pth(folder):
    """
    在 folder 下查找形如 epoch_134_8764.pth 的文件，
    返回 epoch 最大的那个文件的完整路径。
    """
    pattern = re.compile(r"^epoch_(\d+)_(\d+)\.pth$")

    max_epoch = -1
    best_file = None

    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            epoch = int(m.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                best_file = fname

    if best_file is None:
        return None

    return os.path.join(folder, best_file)


def overlap(image1, image2):
    image1 = image1 * 0.7
    rol = np.stack((image1[0, :, :] + image2*2/3, image1[1, :, :], image1[2, :, :]), axis=0)
    rol[rol > 1] = 1
    return rol