import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type


def unit_partition(f: torch.Tensor, s: torch.Tensor):
    """
    Partition feature map `f` into a list of windows (partitions) according to segmentation map `s`.
    Args:
        f: Tensor of shape (B, C, H, W) – input feature map.
        s: Tensor of shape (B, 1, H, W) – segmentation map with integer labels for partitions.
    Returns:
        windows: List of tensors, each of shape (L_i, C), where L_i is the number of pixels in the i-th partition.
        indices: List of tuples (b, idx) mapping each partition window back to the batch index and flattened indices in the original feature map.
    """
    B, C, H, W = f.shape
    windows = []
    indices = []  # to store (batch_index, indices_in_flattened_image) for each partition
    for b in range(B):
        # Flatten feature map and segmentation map for image b
        feat_flat = f[b].view(C, -1)  # shape (C, H*W)
        seg_flat = s[b].view(-1)  # shape (H*W,)
        unique_labels = seg_flat.unique()  # get unique partition labels in this image
        for label in unique_labels:
            # Get all pixel indices belonging to this partition label
            idx = (seg_flat == label).nonzero(as_tuple=False).squeeze(1)  # 1D indices of pixels in this partition
            if idx.numel() == 0:
                continue  # skip if no pixel (should not happen for unique_label)
            # Gather the features for these indices
            region_feats = feat_flat[:, idx]  # shape (C, L) where L is number of pixels in this partition
            # Transpose to shape (L, C) for attention processing
            region_feats_seq = region_feats.t()  # shape (L, C)
            windows.append(region_feats_seq)
            indices.append((b, idx))
    return windows, indices


def unit_unpartition(out_list: list, indices: list, output_shape: torch.Size):
    """
    Reassemble the list of partition outputs back to the original feature map shape.
    Args:
        out_list: List of tensors, each of shape (L_i, C), corresponding to attention outputs for each partition.
        indices: List of tuples (b, idx) as returned by window_partition, indicating where each partition's output should go.
        output_shape: The expected shape of the output feature map (B, C, H, W).
    Returns:
        output: Tensor of shape (B, C, H, W) with all partition outputs merged.
    """
    B, C, H, W = output_shape
    # Initialize an output tensor
    output = torch.zeros((B, C, H * W), device=out_list[0].device, dtype=out_list[0].dtype)
    for out_seq, (b, idx) in zip(out_list, indices):
        # out_seq is shape (L, C) for this partition. Transpose to (C, L) to place into output.
        output[b][:, idx] = out_seq.t()
    # Reshape the output to (B, C, H, W)
    return output.view(B, C, H, W)


def window_partition(f: torch.Tensor, s: torch.Tensor, win_size=32):
    """
    Partition feature map `f` into regular non-overlapping windows.
    Args:
        f: Tensor of shape (B, C, H, W).
        s: Unused placeholder for API compatibility with unit_partition.
        win_size: Window size for both height and width.
    Returns:
        windows: List of tensors, each of shape (L_i, C), where L_i <= win_size * win_size.
        indices: List of tuples (b, idx), where idx are flattened indices in the original HxW map.
    """
    del s  # kept for compatible signature
    B, C, H, W = f.shape
    windows = []
    indices = []

    for b in range(B):
        for h0 in range(0, H, win_size):
            h1 = min(h0 + win_size, H)
            for w0 in range(0, W, win_size):
                w1 = min(w0 + win_size, W)
                window_feat = f[b, :, h0:h1, w0:w1].reshape(C, -1).t().contiguous()

                rows = torch.arange(h0, h1, device=f.device)
                cols = torch.arange(w0, w1, device=f.device)
                rr, cc = torch.meshgrid(rows, cols, indexing="ij")
                idx = (rr * W + cc).reshape(-1)

                windows.append(window_feat)
                indices.append((b, idx))
    return windows, indices

def window_unpartition(out_list: list, indices: list, output_shape: torch.Size):
    """
    Reassemble regular window outputs to the original feature map.
    Args:
        out_list: List of tensors with shape (L_i, C).
        indices: List of tuples (b, idx) from window_partition.
        output_shape: Target shape (B, C, H, W).
    Returns:
        output: Tensor of shape (B, C, H, W).
    """
    B, C, H, W = output_shape
    output = torch.zeros((B, C, H * W), device=out_list[0].device, dtype=out_list[0].dtype)
    for out_seq, (b, idx) in zip(out_list, indices):
        output[b][:, idx] = out_seq.t()
    return output.view(B, C, H, W)


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class SplitAttention(nn.Module):
    def __init__(self, embed_dim=8, num_heads=4, sp=True, sp_mode="unit"):
        """
        Local self-attention module operating within irregular partitions.
        Args:
            embed_dim: Dimension of the feature embeddings (channels of the input feature map).
            num_heads: Number of attention heads for multi-head attention.
        """
        super(SplitAttention, self).__init__()
        # Multi-head attention layer (batch_first=False expects input shape (L, N, E))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=False)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embedding_dim=embed_dim, mlp_dim=int(embed_dim * 4), act=nn.GELU)
        self.sp = sp

        if sp_mode == "unit":
            self.partition = unit_partition
            self.unpartition = unit_unpartition
        elif sp_mode == "window":
            self.partition = window_partition
            self.unpartition = window_unpartition
        else:
            raise NotImplementedError

    def forward(self, f: torch.Tensor, s: torch.Tensor):
        """
        Apply local multi-head self-attention within each partition defined by `s`.
        Args:
            f: Tensor of shape (B, C, H, W) – input feature map.
            s: Tensor of shape (B, 1, H, W) – segmentation map of partitions.
        Returns:
            Tensor of shape (B, C, H, W) – output feature map after local self-attention.
        """
        B, C, H, W = f.shape
        L = H * W
        shortcut = f
        f = self.norm1(f.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        # Partition the feature map into windows based on segmentation
        if self.sp:
            windows, indices = self.partition(f, s)

            out_list = []
            for window_feats in windows:
                # window_feats shape: (L, C) for this partition. Prepare for MultiheadAttention.
                seq = window_feats.unsqueeze(1)  # shape (L, 1, C), treating this partition as a sequence with batch size 1
                attn_out, _ = self.attn(seq, seq, seq)  # self-attention (query, key, value are the same)
                attn_out = attn_out.squeeze(1)  # shape (L, C) after removing batch dim of 1
                out_list.append(attn_out)

            output = self.unpartition(out_list, indices, f.shape)
            output = (shortcut + output)

        else:
            f = f.permute(0, 2, 3, 1).contiguous().reshape(B, -1, C)
            output, _ = self.attn(f, f, f)
            output = shortcut + output.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        output = output + self.mlp(self.norm2(output.permute(0, 2, 3, 1))).permute(0, 3, 1, 2).contiguous()
        return output


def partition_mean_broadcast(
    f: torch.Tensor,  # (B,C,H,W)
    s: torch.Tensor,  # (B,1,H,W)
):
    """
    先按分区求均值 token，再把均值广播回像素，输出仍为 (B,C,H,W)。
    """
    B, C, H, W = f.shape
    new_f = torch.zeros((B, C, H, W), device=f.device, dtype=f.dtype)
    for b in range(B):
        fb = f[b]
        sb = s[b].long()
        for su in sb.unique():
            pos = (sb == su).squeeze()
            mean_v = fb[:, pos].mean(1)
            new_f[b][:, pos] = mean_v[:, None]
    return new_f



if __name__ == "__main__":
    in_f = torch.randn(2, 8, 512, 512).cuda()
    in_s = torch.randint(low=0, high=32, size=(2, 1, 8, 8)).float()
    in_s = F.interpolate(in_s, size=(512, 512), mode="nearest").long().cuda()
    model = SplitAttention(sp=False).cuda()
    out = model(in_f, in_s)
    print(out.shape)
