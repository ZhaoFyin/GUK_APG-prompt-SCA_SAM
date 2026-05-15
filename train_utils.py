import os
import torch
import torch.nn as nn
from utils import Accuracy
from losses import UnetFormerLoss
import gc

import numpy as np
import torch.optim as optim

from tqdm import tqdm


def val(model, dataloader, multimask_output):
    val_accuracy = Accuracy()
    t = tqdm(dataloader, desc=f"val...", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for sampled_batch in t:
            image_batch, label_batch, k_batch, s_batch, p_batch = sampled_batch

            image_batch, label_batch, k_batch, s_batch = image_batch.cuda(), label_batch.cuda(), k_batch.cuda(), s_batch.cuda()
            outputs, prompts = model(image_batch, k_batch, s_batch, p_batch)
            out_logits = torch.sigmoid(outputs['masks'])
            pre_mask = (out_logits > 0.5) * 1.0
            val_accuracy.update(pre_mask, label_batch)

        pa, mpa, miou = val_accuracy.calculate_metrics()
    t.clear()
    t.close()
    val_accuracy.reset()
    return pa, mpa, miou


def trainer_synapse(args, model, snapshot_path, trainloader, valloader, multimask_output, CLASSES):
    model.train()
    base_lr = args.base_lr
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr

    # ======== 参数分组：LoRA 参数单独设学习率 ========
    lora_params, base_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "A" in name or "B" in name:
            lora_params.append(p)
        else:
            base_params.append(p)

    param_groups = [
        {"params": base_params, "lr": b_lr},  # 正常参数
        {"params": lora_params, "lr": b_lr * args.lora_cfg["lr_rate"]},
    ]

    # ======== 选择优化器 ========
    if args.AdamW:
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=0.0001
        )

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    print("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_miou = 0.0
    criterion = UnetFormerLoss(num_c=1)
    lora_rate = float(args.lora_cfg["lr_rate"])
    train_accuracy = Accuracy()

    for epoch_num in range(max_epoch):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=100,
                    desc=f"Epoch [{epoch_num + 1}/{max_epoch}]", leave=False)
        train_accuracy.reset()
        for iter_idx, sampled_batch in pbar:

            image_batch, label_batch, k_batch, s_batch, p_batch = sampled_batch

            image_batch, label_batch, k_batch, s_batch = image_batch.cuda(), label_batch.cuda(), k_batch.cuda(), s_batch.cuda()
            outputs, prompts = model(image_batch, k_batch, s_batch, p_batch)
            logits = outputs['masks']
            pre_mask = (torch.sigmoid(logits) > 0.5) * 1.0

            loss = criterion(logits, label_batch.long())

            train_accuracy.update(pre_mask, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            epoch_loss += loss.item()
            if args.warmup and iter_num < args.warmup_period:
                # 线性 warmup：随迭代从 0 → base_lr
                base_lr_t = args.base_lr * ((iter_num + 1) / args.warmup_period)
            else:
                # Poly（或 Cosine）衰减，这里沿用你原来的 poly^0.9
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f"Shift iter {shift_iter} < 0"
                    total_after = max(1, max_iterations - args.warmup_period)
                    decay_pos = min(shift_iter, total_after)
                    frac = 1.0 - decay_pos / total_after
                else:
                    decay_pos = min(iter_num, max_iterations)
                    frac = 1.0 - decay_pos / max_iterations
                base_lr_t = args.base_lr * (frac ** 0.9)

            if len(optimizer.param_groups) >= 1:
                optimizer.param_groups[0]["lr"] = base_lr_t

            if len(optimizer.param_groups) >= 2:
                optimizer.param_groups[1]["lr"] = base_lr_t * lora_rate
            lr_base_show = optimizer.param_groups[0]["lr"]
            lr_lora_show = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else -1.0

            pbar.set_postfix({
                "ls": f"{loss.item():.4f}",
                "lr": f"{lr_base_show:.6f}/{lr_lora_show:.6f}"
            })
        epoch_loss = epoch_loss / len(trainloader)
        pa, mpa, miou = train_accuracy.calculate_metrics()
        train_value = {'pa': pa,
                       'mpa': mpa,
                       'miou': miou}
        print("Epoch: [{}/{}]\t Loss: {}".format(epoch_num + 1, max_epoch, epoch_loss))
        print('train:', train_value)
        pbar.clear()
        pbar.close()

        val_pa, val_mpa, val_miou = val(model, valloader, multimask_output)

        val_value = {'pa': val_pa,
                       'mpa': val_mpa,
                       'miou': val_miou}
        print('val:', val_value)
        # val
        if val_miou > best_miou:
            best_miou = val_miou
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num+1) + "_" + str(int(10000*val_miou)) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            print("\t save to {}".format(save_mode_path))

    return "Training Finished!"
