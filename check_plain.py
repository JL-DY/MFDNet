import os
import cv2 as cv
import torch
import numpy as np
import torch.nn as nn
import time
from model.MFDNet import MFDNet
from model.Plaindn import PlainDN
from train import opt
from utils.Rep_params import rep_params

def main():
    # set hardware specification
    if opt.cuda >=0 and torch.cuda.is_available():
        print("use cuda & cudnn for acceleration!")
        print("the gpu id is:{}".format(opt.cuda))
        device = torch.device('cuda:{}'.format(opt.cuda))
    else:
        print("use cpu for training or testing")
        device = torch.device('cpu')


    model = MFDNet(opt.M_MFDB, opt.K_RepConv, opt.c_channel).to(device)
    model_plain = PlainDN(opt.M_MFDB, opt.K_RepConv, opt.c_channel).to(device)
    print("load model: {}!".format(opt.test_model))
    model.load_state_dict(torch.load(opt.test_model))

    # 参数重构化
    model_plain = rep_params(model, model_plain, opt, device)

    x = torch.randn(1, 1, 24, 24).to(device)
    result = model_plain(x)
    result1 = model(x)
    print(result - result1)


if __name__ == "__main__":
    main()
    exit(0)
    