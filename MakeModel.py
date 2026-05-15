import torch
import torch.nn as nn
import torch.nn.functional as F
from KgeModel.window_attention import *
from KgeModel.KgeBlock import KnowledgeOut
from SAM.build_sam import sam_model_registry

def remap_for_lora(w):
    new_w = {}
    for k, v in list(w.items()):
        if "qkv" in k:
            new_k = k.replace(".weight", ".base.weight")
            new_w[new_k] = v
        elif "proj" and ".weight" in k:
            new_k = k.replace(".weight", ".base.weight")
            new_w[new_k] = v
        elif "proj" and ".bias" in k:
            new_k = k.replace(".bias", ".base.bias")
            new_w[new_k] = v
        else:
            new_w[k] = v
    return new_w


class KgeScaSam(nn.Module):
    def __init__(self, lora_cfg, num_classes, sca, num_k,
                 threshold=None, img_s=512,
                 pmt_mode="mask", pmt_set="auto", post_mode="unit"):
        super(KgeScaSam, self).__init__()
        self.pmt_mode = pmt_mode
        self.pmt_set = pmt_set
        if self.pmt_mode == "mask" and self.pmt_set == "auto":
            # self.prompt_gen = KnowledgeOut(num_k=num_k, lack=None)
            assert post_mode in ["unit", "window", None]
            post_process = None
            if post_mode is not None:
                post_process = nn.ModuleList([SplitAttention(sp=True, sp_mode=post_mode),
                                              SplitAttention(sp=True, sp_mode=post_mode)])
            self.prompt_gen = KnowledgeOut(num_k=num_k, lack=None,
                                           post_process=post_process, sp_mode=post_mode)
            if post_mode is not None:
                w = torch.load(r"pretrain_weight/epoch_64_6907.pth")
                new_w = {}
                for k, v in list(w.items()):
                    new_w[k.replace("prompt_gen.", "")] = v
                self.prompt_gen.load_state_dict(new_w)
                self.prompt_gen.requires_grad = False

            self.threshold = threshold

        self.img_s = img_s

        self.sam, _ = sam_model_registry["vit_b"](image_size=self.img_s,
                                                  num_classes=num_classes,
                                                  checkpoint=r"./pretrain_weight/sam_vit_b_01ec64.pth",
                                                  pixel_mean=[0.3394, 0.3598, 0.3226],
                                                  pixel_std=[0.2037, 0.1899, 0.1922],
                                                  use_sca=sca,
                                                  lora_cfg=lora_cfg,
                                                  process_cp=lora_cfg["enable"],
                                                  report=False)
        if sca:
            sca_w = torch.load(r"pretrain_weight/sca.pth")
            if lora_cfg["sca_enable"]:
                sca_w = remap_for_lora(sca_w)

            self.sam.load_state_dict(sca_w, strict=False)


        for n, p in self.sam.image_encoder.named_parameters():
            p.requires_grad = False
            if hasattr(p, "is_lora_param"):
                p.requires_grad = True
            if ".sca." in n:
                p.requires_grad = True
            if "base" in n:
                p.requires_grad = False
            if "neck" in n:
                p.requires_grad = False

        self.sam.mask_decoder.requires_grad = True

    def forward(self, x, y, s=None, pmt=None, multi_output=False, img_size=512):
        sam_prompt = {"mask": [None, None], "point": [None, None], "box": [None, None]}
        if self.pmt_mode == "mask" and self.pmt_set == "auto":
            _, prompt = self.prompt_gen(y,s)
            prompt = F.interpolate(prompt, (self.img_s//4, self.img_s//4), mode="bilinear", align_corners=False)
            if self.threshold is not None:
                prompt = prompt[:, 1]
                prompt[prompt < self.threshold] = 0
            else:
                prompt = F.sigmoid(prompt[:, 1] - prompt[:, 0])
            sam_prompt["mask"] = prompt
        elif self.pmt_mode == "point":
            prompt = pmt[self.pmt_set]
            prompt = [[prompt[0][i], prompt[1][i]] for i in range(len(prompt[0]))]
            sam_prompt["point"] = prompt
        elif self.pmt_mode == "box":
            prompt = pmt[self.pmt_set]
            prompt = [[prompt[0][i], prompt[1][i], prompt[2][i], prompt[3][i]] for i in range(len(prompt[0]))]
            sam_prompt["box"] = prompt
        else:
            raise NotImplementedError


        outputs = self.sam(x, sam_prompt, multi_output, img_size)
        return outputs, sam_prompt

