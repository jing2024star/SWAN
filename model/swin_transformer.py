import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['SwinTransformerBlock']

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (3, B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # Attention scores

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # Output from attention
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))

    
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        # 第一个子块: W-MSA
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, attn_drop=attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        # 第二个子块: SW-MSA (仅当 shift_size > 0 时)
        if self.shift_size > 0:
            self.norm3 = nn.LayerNorm(dim)
            self.attn_shift = WindowAttention(dim, window_size, num_heads, attn_drop=attn_drop)
            self.norm4 = nn.LayerNorm(dim)
            self.mlp_shift = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward_block(self, x, H, W, shift_size, attn_layer, norm_layer1, norm_layer2, mlp_layer):
        B, N, C = x.shape
        shortcut = x
        x = norm_layer1(x)
        x = x.view(B, H, W, C)

        window_size = min(self.window_size, H, W)
        actual_shift = shift_size if window_size > shift_size else 0

        # 处理填充
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H_padded, W_padded = H + pad_h, W + pad_w
        else:
            H_padded, W_padded = H, W

        # 循环移位
        if actual_shift > 0:
            shifted_x = torch.roll(x, shifts=(-actual_shift, -actual_shift), dims=(1, 2))
            # 计算掩码
            img_mask = torch.zeros((1, H_padded, W_padded, 1), device=x.device)
            h_slices = (slice(0, -window_size),
                        slice(-window_size, -actual_shift),
                        slice(-actual_shift, None))
            w_slices = (slice(0, -window_size),
                        slice(-window_size, -actual_shift),
                        slice(-actual_shift, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_x = x
            attn_mask = None

        # 分割窗口
        x_windows = window_partition(shifted_x, window_size)
        x_windows = x_windows.view(-1, window_size * window_size, C)

        # 窗口注意力
        attn_windows = attn_layer(x_windows, mask=attn_mask)

        # 反转分割窗口
        attn_windows = attn_windows.view(-1, window_size, window_size, C)
        shifted_x = window_reverse(attn_windows, window_size, H_padded, W_padded)

        # 反转循环移位
        if actual_shift > 0:
            x = torch.roll(shifted_x, shifts=(actual_shift, actual_shift), dims=(1, 2))
        else:
            x = shifted_x

        # 移除填充
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, N, C)
        x = shortcut + x

        # MLP 层
        x = x + mlp_layer(norm_layer2(x))
        return x

    def forward(self, x):
        B, N, C = x.shape
        # 自动计算正确的分辨率
        H = int(math.sqrt(N))
        W = H
        if H * W != N:
            H = int(math.sqrt(N))
            W = N // H
            while H * W != N and H > 1:
                H -= 1
                W = N // H

        # 1. 运行 W-MSA (shift_size=0)
        x = self.forward_block(x, H, W, 0, self.attn, self.norm1, self.norm2, self.mlp)

        # 2. 运行 SW-MSA (shift_size=self.shift_size)
        if self.shift_size > 0:
            x = self.forward_block(x, H, W, self.shift_size, self.attn_shift, self.norm3, self.norm4, self.mlp_shift)

        return x