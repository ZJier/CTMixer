import einops
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class Attention_Conv(nn.Module):

    def __init__(self, dim: int, head_dim: int, num_heads: int, num_patch: int, patch_size: int):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.inner_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5
        self.attn = nn.Softmax(dim=-1)
        self.act = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(dim)
        self.qkv = nn.Conv2d(dim, self.inner_dim * 3, kernel_size=1, padding=0, groups=dim, bias=False)
        self.avgpool=nn.AdaptiveAvgPool1d(dim)

        self.qs = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim, bias=False)
        self.ks = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        x = x.contiguous().view(b, self.dim, self.num_patch, self.num_patch)

        qkv = self.qkv(self.act(self.bn(x)))
        qkv = qkv.contiguous().view(b, self.num_patch*self.num_patch, self.inner_dim * 3)
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)

        q_1 = q[:, 0:self.num_heads//2, :, :]
        k_1 = k[:, 0:self.num_heads//2, :, :]
        v_1 = v[:, 0:self.num_heads//2, :, :]
        q_2 = q[:, self.num_heads//2:self.num_heads, :, :].reshape(b, -1, int(math.sqrt(n)), int(math.sqrt(n)))
        k_2 = k[:, self.num_heads//2:self.num_heads, :, :].reshape(b, -1, int(math.sqrt(n)), int(math.sqrt(n)))
        v_2 = v[:, self.num_heads//2:self.num_heads, :, :].reshape(b, -1, int(math.sqrt(n)), int(math.sqrt(n)))

        q_2 = self.qs(q_2)
        k_2 = self.ks(k_2)
        res_2 = (q_2 + k_2 + v_2).reshape(b, n, -1)

        scores = torch.einsum("b h i d, b h j d -> b h i j", q_1, k_1)
        scores = scores * self.scale
        attn = self.attn(scores)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v_1)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        res = torch.cat([out, res_2], axis=2)
        out = self.avgpool(res)
        return out
    

class FeedForward_Conv(nn.Module):

    def __init__(self, dim: int, hidden_dim: int, num_patch: int, patch_size: int):
        super().__init__()
        self.dim = dim
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(dim), nn.GELU(), 
            nn.Conv2d(dim, 64, kernel_size=1, padding=0, bias=False))

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(64), nn.GELU(), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False))
        
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(64), nn.GELU(), 
            nn.Conv2d(64, dim, kernel_size=1, padding=0, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, hw, dim = x.shape     # [bs, num_seq, dim]
        x_reshape = x.contiguous().view(b, self.dim, self.num_patch, self.num_patch)
        out1 = self.conv1(x_reshape)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2) + x_reshape
        result = out3.contiguous().view(b, self.num_patch * self.num_patch, self.dim)

        return result


class transformer(nn.Module):

    def __init__(self, dim: int, num_layers: int, num_heads: int, head_dim: int, hidden_dim: int, num_patch: int, patch_size: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = [
                nn.Sequential(nn.LayerNorm(dim), Attention_Conv(dim, head_dim, num_heads, num_patch, patch_size)),
                nn.Sequential(nn.LayerNorm(dim), FeedForward_Conv(dim, hidden_dim, num_patch, patch_size))
            ]
            self.layers.append(nn.ModuleList(layer))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x