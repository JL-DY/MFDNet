import torch 
import torch.nn as nn
import onnx
import numpy as np
from model.MFDNet import MFDNet
from model.ECBSR_MFDNet import ECBSR_MFDNET, ECBSR_MFDNET_PLAIN
from train import opt

if __name__=="__main__":
    if opt.cuda >=0 and torch.cuda.is_available():
        print("use cuda & cudnn for acceleration!")
        print("the gpu id is:{}".format(opt.cuda))
        device = torch.device('cuda:{}'.format(opt.cuda))
    else:
        print("use cpu for training or testing")
        device = torch.device('cpu')

    # 仅转换mfdnet模型以及转换ecbsr+mfdnet模型
    only_mfdnet = True
    if only_mfdnet:
        # 加载模型
        model = MFDNet(opt.M_MFDB, opt.K_RepConv, opt.c_channel).to(device)
        model.load_state_dict(torch.load(opt.test_model))
        # 设置为eval模式，固定bn等操作
        model.eval()
        model.to("cpu")
        # 设置模型的输入
        input = torch.randn((1, 1, 90, 120))
        torch.onnx.export(model, input, "./mfdnet-sigma8.onnx", input_names=["inputs"], output_names=["outputs"], 
                        dynamic_axes={"inputs":{2:"inputs_height", 3:"inputs_weight"}, "outputs":{2:"outputs_height", 3:"outputs_weight"}})
        print("Model has benn converted to onnx")

    else:
        model_ecbsr_mfdnet = ECBSR_MFDNET_PLAIN(4, 16,'prelu', 4, 1, 3, 3, 16).to(device)
        model_ecbsr_mfdnet.load_state_dict(torch.load("./weights/ecbsr_mfdnet_plain/ecbsr_mfdnet_plain.pt"))
        model_ecbsr_mfdnet.eval()
        model_ecbsr_mfdnet.to("cpu")

        input = torch.randn((1, 1, 90, 120))
        torch.onnx.export(model_ecbsr_mfdnet, input, "./ecbsr_mfdnet_plain-sigma8.onnx", input_names=["inputs"], output_names=["outputs"], 
                        dynamic_axes={"inputs":{2:"inputs_height", 3:"inputs_weight"}, "outputs":{2:"outputs_height", 3:"outputs_weight"}})
        print("Model has benn converted to onnx")
