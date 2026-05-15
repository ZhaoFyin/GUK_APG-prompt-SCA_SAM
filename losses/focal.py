import torch
import torch.nn.functional as F

def focal_loss(inputs, targets, alpha=5, gamma=2.0):
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-bce_loss)  # pt 是预测为正类的概率
    f_loss = alpha * (1 - pt) ** gamma * bce_loss
    return f_loss.mean()