import torch
import torch.nn as nn
from .Haar_trans import haar_trans, Upsampling
import torch.nn.functional as F

class RepConv(nn.Module):
    def __init__(self, c):
        super(RepConv, self).__init__()
        self.repconv = nn.Sequential(nn.Conv2d(c, 2*c, 1),
                                     nn.Conv2d(2*c, 2*c, 3, padding=1),
                                     nn.Conv2d(2*c, c, 1))
        
    def forward(self, x):
        repconv_residual = self.repconv(x) + x 
        return repconv_residual

class Interpolate(nn.Module):
    def __init__(self, scale):
        super(Interpolate, self).__init__()
        self.scale = scale

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
    
class MFA(nn.Module):
    def __init__(self, c):
        super(MFA, self).__init__()
        self.input_c = 4*c #haar变换做降采样需*4
        self.output_c = c
        self.mfa = nn.Sequential(haar_trans(),
                                 nn.Conv2d(self.input_c, self.output_c, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(self.output_c, self.output_c, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(self.output_c, self.output_c, 3, padding=1),
                                 nn.Sigmoid(),
                                 Interpolate(2))
    
    def forward(self, x):
        mfa_residual = self.mfa(x) * x
        return mfa_residual
    

class MFDB(nn.Module):
    def __init__(self, k, c):
        super(MFDB, self).__init__()
        self.k_rep = k
        self.c_channel = c

        mfdb = []
        for i in range(self.k_rep):
            mfdb += [RepConv(self.c_channel)]
        mfdb += [MFA(self.c_channel)]
        self.mfdb = nn.Sequential(*mfdb)

    def forward(self, x):
        mfdb = self.mfdb(x) + x # 应该没有跳跃链接才对，但是目前效果还阔以，就留着
        return mfdb

class MFDB_Plain(nn.Module):
    def __init__(self, k, c):
        super(MFDB_Plain, self).__init__()
        self.k_rep = k
        self.c_channel = c

        mfdb = []
        for i in range(self.k_rep):
            mfdb += [nn.Conv2d(self.c_channel, self.c_channel, 3, padding=1)]
        mfdb += [MFA(self.c_channel)]
        self.mfdb = nn.Sequential(*mfdb)

    def forward(self, x):
        mfdb = self.mfdb(x) + x
        return mfdb
