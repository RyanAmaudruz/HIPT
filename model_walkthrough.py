import sys

sys.path.extend([
    './1-Hierarchical-Pretraining/', '2-Weakly-Supervised-Subtyping', 'HIPT_4K'
])


import torch
import torch.nn.functional as F

from models.model_hierarchical_mil import HIPT_LGP_FC

## 1. Example Forward Pass (with Pre-Extracted x_256 Features)

x = torch.randn(38,256,384)
self = HIPT_LGP_FC()
self.forward(x)

## 2. Forward Pass Shape Walkthrough (with Pre-Extracted x_256 Features)


x_256 = torch.randn(38,256,384)
print("1. Input Tensor:", x_256.shape)
print()
x_256 = x_256.unfold(1, 16, 16).transpose(1,2)
print("2. Re-Arranging 1D-(Seq Length of # [256x256] tokens in [4096x4096] Region) Axis to be a 2D-Grid:", x_256.shape)
print()

h_4096 = self.local_vit(x_256)
print("3. Seq length of [4096x4096] Tokens in the WSI", h_4096.shape)
print()

h_4096 = self.global_phi(h_4096)
h_4096 = self.global_transformer(h_4096.unsqueeze(1)).squeeze(1)
A_4096, h_4096 = self.global_attn_pool(h_4096)
A_4096 = torch.transpose(A_4096, 1, 0)
A_4096 = F.softmax(A_4096, dim=1)
h_path = torch.mm(A_4096, h_4096)
h_WSI = self.global_rho(h_path)
print("4. ViT-4K + Global Attention Pooling to get WSI-Level Embedding:", h_WSI.shape)

