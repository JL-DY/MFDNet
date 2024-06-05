import torch
import torch.nn as nn
import torch.nn.functional as F
from .Haar_trans import haar_trans, Upsampling
from .MFDB import MFDB, MFDB_Plain


class PlainDN(nn.Module):
    def __init__(self, m, k, c):
        super(PlainDN, self).__init__()
        self.m_mfdb = m
        self.k_rep = k
        self.c_channel = c
        
        # harr transform做x4降采样
        self.downsample_blocks = nn.Sequential(haar_trans(),
                                               haar_trans())
        
        # m MFDB modules are created
        M_block = []
        for i in range(self.m_mfdb):
            M_block += [MFDB_Plain(self.k_rep, self.c_channel)]
        self.M_block = nn.Sequential(*M_block)

        # reconstruction
        self.recon = nn.Sequential(nn.Conv2d(self.c_channel, self.c_channel, 3, padding=1),
                                   nn.PixelShuffle(4))

    def forward(self, x):
        haar = self.downsample_blocks(x)
        mfdb = self.M_block(haar) + haar
        mfdnet = self.recon(mfdb) + x
        return mfdnet
