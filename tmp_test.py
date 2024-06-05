import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


# 创建一个1x1的卷积层，使用单位矩阵作为初始化权重
conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, bias=False)
with torch.no_grad():
    # 将权重设置为单位矩阵
    conv.weight.data.copy_(torch.eye(16).view(16, 16, 1, 1))  # 假设输入输出通道数为64
pad_dims = (1, 1, 1, 1, 0, 0, 0, 0)
x = torch.nn.functional.pad(conv.weight.data, pad=pad_dims, mode='constant', value=0)
a = torch.rand((16, 16, 24, 24))
b = F.conv2d(a, x, padding=1)
equal_tenor = torch.eq(a, b)
are_equal = equal_tenor.all().item()
print(are_equal)
