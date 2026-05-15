import torch
import torch.nn as nn
from dataset import CustomDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import Accuracy, overlap
from torchvision.transforms import transforms, InterpolationMode
from KgeModel.KgeBlock import KnowledgeOut
from KgeModel.window_attention import *
from tqdm import tqdm
from losses import UnetFormerLoss, focal_loss
import os
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

class KgeScaSam(nn.Module):
    def __init__(self, num_k, img_s=512):
        super(KgeScaSam, self).__init__()

        self.prompt_gen = KnowledgeOut(num_k=num_k, lack=None,
                                       post_process=nn.ModuleList([SplitAttention(sp=True),
                                                                   SplitAttention(sp=True),]))

        self.img_s = img_s
        w = torch.load(r"./pretrain_weight/org.pt")
        self.prompt_gen.load_state_dict(w, strict=False)
        self.prompt_gen.requires_grad_(False)
        self.prompt_gen.post_process.requires_grad_(True)

    def forward(self, y, s):
        _, prompt = self.prompt_gen(y, s)
        # prompt = F.interpolate(prompt, (self.img_s//4, self.img_s//4), mode="bilinear", align_corners=False)
        # output = prompt[:,1].unsqueeze(1) - prompt[:,0].unsqueeze(1)
        return prompt

dataset_name = "YYL"
data_dict = {"YYL": r"G:\KGE-SwinFpn/VOCdevkit_YYL",
             "BJL": r"G:\KGE-SwinFpn/VOCdevkit_BJL"}
data_path = data_dict[dataset_name]

def train():
    max_epoch = 100
    model = KgeScaSam(num_k=9, img_s=512)

    model.cuda()

    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset(voc_root=data_path, kge=True, txt_name="train.txt",
                                  transform=transform, oh=True)
    val_dataset = CustomDataset(voc_root=data_path, kge=True, txt_name="val.txt",
                                transform=transform, oh=True)


    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=2,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False)
    iter_num = 0
    best_miou = 0.0
    model.train()
    criterion = focal_loss
    # criterion = UnetFormerLoss(num_c=1)
    train_accuracy = Accuracy()
    val_accuracy = Accuracy()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0006)
    for epoch_num in range(max_epoch):
        epoch_loss = 0.0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100,
                    desc=f"Epoch [{epoch_num + 1}/{max_epoch}]", leave=False)

        for iter_idx, sampled_batch in pbar:
            image_batch, label_batch, k_batch, s_batch = sampled_batch

            image_batch, label_batch, k_batch, s_batch = image_batch.cuda(), label_batch.cuda(), k_batch.cuda(), s_batch.cuda()
            outputs = model(k_batch, s_batch)
            loss = criterion(outputs, label_batch.float())

            pre_mask = torch.argmax(outputs, dim=1)
            labels = torch.argmax(label_batch, dim=1)
            train_accuracy.update(pre_mask, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            epoch_loss += loss.item()
            pbar.set_postfix({"ls": f"{loss.item():.4f}"})
        pbar.clear()
        pbar.close()
        epoch_loss = epoch_loss / len(train_loader)
        pa, mpa, miou = train_accuracy.calculate_metrics()
        train_accuracy.reset()
        train_value = {'pa': pa,
                       'mpa': mpa,
                       'miou': miou}
        print("Epoch: [{}/{}]\t Loss: {}".format(epoch_num + 1, max_epoch, epoch_loss))
        print('train:', train_value)


        t = tqdm(val_loader, desc=f"val...", leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for sampled_batch in t:
                image_batch, label_batch, k_batch, s_batch = sampled_batch

                image_batch, label_batch, k_batch, s_batch = image_batch.cuda(), label_batch.cuda(), k_batch.cuda(), s_batch.cuda()
                outputs = model(k_batch, s_batch)

                pre_mask = torch.argmax(outputs, dim=1)
                labels = torch.argmax(label_batch, dim=1)
                val_accuracy.update(pre_mask, labels)

            t.clear()
            t.close()
            val_pa, val_mpa, val_miou = val_accuracy.calculate_metrics()
            val_accuracy.reset()
            val_value = {'pa': val_pa,
                         'mpa': val_mpa,
                         'miou': val_miou}

        print('val:', val_value)
        if val_miou > best_miou:
            best_miou = val_miou
            save_mode_path = os.path.join("pretrain_weight",
                                          'epoch_' + str(epoch_num + 1) + "_" + str(int(10000 * val_miou)) + '.pth')
            torch.save(model.state_dict(), save_mode_path)

            print("\t save to {}".format(save_mode_path))
    return "finished"

def test():
    model = KgeScaSam(num_k=9, img_s=512)
    model.load_state_dict(torch.load("pretrain_weight/epoch_64_6907.pth"))
    model.cuda()

    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    os.makedirs("demo", exist_ok=True)
    test_dataset = CustomDataset(voc_root=data_path, kge=True, txt_name="val.txt",
                                transform=transform, oh=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)
    t = tqdm(test_loader, desc=f"val...", leave=False, dynamic_ncols=True)
    accuracy = Accuracy()
    with torch.no_grad():
        _num = 0
        for sampled_batch in t:
            image_batch, label_batch, k_batch, s_batch = sampled_batch

            image_batch, label_batch, k_batch, s_batch = image_batch.cuda(), label_batch.cuda(), k_batch.cuda(), s_batch.cuda()
            outputs = model(k_batch, s_batch)
            pre_mask = torch.argmax(outputs, dim=1)
            labels = torch.argmax(label_batch, dim=1)
            accuracy.update(pre_mask, labels)
            for image, label, pred in zip(image_batch, labels, pre_mask):
                sub_a = overlap(image.cpu().numpy(), label.squeeze().cpu().numpy()).transpose(1, 2, 0)

                sub_c = overlap(image.cpu().numpy(), pred.squeeze().cpu().numpy()).transpose(1, 2, 0)
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].imshow(sub_a)
                ax[0].set_title("True")

                ax[1].imshow(sub_c)
                ax[1].set_title("Pred")
                fig.savefig(f"demo/{_num}.png")
                _num += 1
                pass
        t.clear()
        t.close()
        val_pa, val_mpa, val_miou = accuracy.calculate_metrics()
        print("pa: {}, mpa: {}, miou: {}".format(val_pa, val_mpa, val_miou))

if __name__ == '__main__':
    # train()
    test()