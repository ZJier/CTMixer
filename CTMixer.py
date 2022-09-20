# Torch
import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from torchinfo import summary
import torch.nn.functional as F
from transformer import transformer
from embeddings import (PatchEmbeddings, CLSToken, PositionalEmbeddings)


class Pooling(nn.Module):
    """
    @article{ref-vit,
	title={An image is worth 16x16 words: Transformers for image recognition at scale},
	author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, 
            Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
	journal={arXiv preprint arXiv:2010.11929},
	year={2020}
    }
    """
    def __init__(self, pool: str = "mean"):
        super().__init__()
        if pool not in ["mean", "cls"]:
            raise ValueError("pool must be one of {mean, cls}")

        self.pool_fn = self.mean_pool if pool == "mean" else self.cls_pool

    def mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def cls_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_fn(x)


class Classifier(nn.Module):

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Res2(nn.Module):  
    """
    @article{ref-sprn,
	title={Spectral partitioning residual network with spatial attention mechanism for hyperspectral image classification},
	author={Zhang, Xiangrong and Shang, Shouwang and Tang, Xu and Feng, Jie and Jiao, Licheng},
	journal={IEEE Trans. Geosci. Remote Sens.},
	volume={60},
	pages={1--14},
	year={2021},
	publisher={IEEE}
    }
    """
    def __init__(self, in_channels, inter_channels, kernel_size, padding=0):
        super(Res2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, in_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.bn2(self.conv2(X))
        return X


class Res(nn.Module):  
    def __init__(self, in_channels, kernel_size, padding, groups_s):
        super(Res, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups_s)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups_s)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.res2 = Res2(in_channels, 32, kernel_size=kernel_size, padding=padding)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Z = self.res2(X)
        return F.relu(X + Y + Z)


class CTMixer(nn.Module):

    def __init__(self, channels, num_classes, image_size, datasetname, num_layers: int=1, num_heads: int=4, 
                 patch_size: int = 1, emb_dim: int = 128, head_dim = 64, hidden_dim: int = 64, pool: str = "mean"):
        super().__init__()
        self.emb_dim = emb_dim

        self.hidden_dim = hidden_dim
        self.channels = channels
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_patch = int(math.sqrt(self.num_patches))
        self.act = nn.ReLU(inplace=True)
        patch_dim = channels * patch_size ** 2

        # Conv Preprocessing Module (Ref-SPRN)
        if datasetname == 'IndianPines':
            groups = 11
            groups_width = 37
        elif datasetname == 'PaviaU':
            groups = 5
            groups_width = 64
        elif datasetname == 'Salinas':
            groups = 11
            groups_width = 37
        elif datasetname == 'Houston':
            groups = 5
            groups_width = 64
        else:
            groups = 11
            groups_width = 37
        new_bands = math.ceil(channels/groups) * groups
        patch_dim = (groups*groups_width) * patch_size ** 2
        pad_size = new_bands - channels
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, pad_size))
        self.conv_1 = nn.Conv2d(new_bands, groups*groups_width, (1, 1), groups=groups)
        self.bn_1 = nn.BatchNorm2d(groups*groups_width)
        self.res0 = Res(groups*groups_width, (3, 3), (1, 1), groups_s=groups)

        # # Dual Residual Block (Ref-RDACN (mine))
        self.bn1 = nn.BatchNorm2d(emb_dim)
        self.conv1 = nn.Conv2d(emb_dim, 64, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, emb_dim, kernel_size=1, padding=0)

        # Vision Transformer
        self.patch_embeddings = PatchEmbeddings(patch_size=patch_size, patch_dim=patch_dim, emb_dim=emb_dim)
        self.pos_embeddings = PositionalEmbeddings(num_pos=self.num_patches, dim=emb_dim)
        self.transformer = transformer(dim=emb_dim, num_layers=num_layers, num_heads=num_heads, 
                                        head_dim=head_dim, hidden_dim=hidden_dim, num_patch=self.num_patch, patch_size=patch_size)
        self.dropout = nn.Dropout(0.5)

        self.pool = Pooling(pool=pool)
        self.classifier = Classifier(dim=emb_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x).squeeze(1)
        b, c, h, w = x.shape
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = self.res0(x)

        x4 = self.patch_embeddings(x)
        x5 = self.pos_embeddings(x4)
        x6 = self.transformer(x5)

        x4_c = x4.reshape(b, -1, h, w)
        x_c1 = self.conv1(self.act(self.bn1(x4_c)))
        x_c2 = self.conv2(self.act(self.bn2(x_c1)))
        x_c3 = self.conv3(self.act(self.bn3(x_c2)))

        x7 = self.pool(self.dropout(x6 + x_c3.reshape(b, h*w, -1)))

        return self.classifier(x7)


if __name__ == '__main__':
    input = torch.randn(size=(100, 1, 200, 11, 11))
    print("input shape:", input.shape)
    model = CTMixer(channels=200, num_classes=16, image_size=11, datasetname='IndianPines', num_layers=1, num_heads=4)
    summary(model, input_size=(100, 1, 200, 11, 11), device="cpu")
    print("output shape:", model(input).shape)
