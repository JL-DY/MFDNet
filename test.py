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
    
    model.eval()
    model_plain.eval()
    process_video = False

    if (process_video):
        # start video process
        video_capture = cv.VideoCapture('./test_video2.mp4')

        if not video_capture.isOpened():
            print("无法打开视频文件")
            exit()

        # 获取视频的基本信息
        frame_width = int(video_capture.get(3))
        frame_height = int(video_capture.get(4))
        fps = int(video_capture.get(5))

        # 创建一个VideoWriter对象，用于保存处理后的视频
        out = cv.VideoWriter('./output_video2.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height), isColor=True)

        while True:
            ret, frame = video_capture.read()

            if not ret:
                break

            frame = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
            y_channel = frame[:, :, 0]
            crcb_channel = frame[:, :, 1:]
            y_channel = cv.resize(y_channel, (90, 120))
            y_channel = np.array(y_channel, dtype=np.float32)
            y_channel = np.expand_dims(np.expand_dims(y_channel, axis=0), axis=0)
            y_channel = torch.tensor(y_channel).to(device)
            with torch.no_grad():
                result = model_plain(y_channel)
            
            result = result[0].cpu().numpy()
            result = np.clip(result, 0, 255)
            result = np.array(result, dtype=np.uint8)
            result = np.transpose(result, (1, 2, 0))
            result = cv.resize(result, (frame_width, frame_height))
            out_result = cv.merge([result, crcb_channel], -1)
            out_result = cv.cvtColor(out_result, cv.COLOR_YCrCb2BGR)
            # 写处理后的帧到输出视频文件
            out.write(out_result)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        out.release()
    else:
        if not os.path.exists(opt.test_output):
            os.makedirs(opt.test_output)
        img_name = os.listdir(opt.test_input)
        for i in img_name:
            img = cv.imread(os.path.join(opt.test_input, i))
            img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
            CrCb = img[:, :, 1:]
            Y_channel = img[:, :, 0]
            Y_channel = np.array(np.expand_dims(np.expand_dims(Y_channel, axis=0), axis=0), dtype=np.float32) / 255
            Y_channel = torch.tensor(Y_channel).to(device)
            
            with torch.no_grad():
                time_start = time.time()
                result = model(Y_channel)
                print("use time:", time.time()-time_start)
            result = result[0].cpu().numpy()
            result = np.clip(result, 0, 1) * 255
            result = np.array(result, dtype=np.uint8)
            result = np.transpose(result, (1, 2, 0))
            
            new_result = cv.merge((result, CrCb))
            new_result = cv.cvtColor(new_result, cv.COLOR_YCrCb2BGR)
            cv.imwrite(os.path.join(opt.test_output, i), new_result)
            
        # 保存参数重构化后的模型 
        # torch.save(model_plain.state_dict(), "./weights/mfdnet_plain/mfdnet_plain_sigma11.pt")
        
            
if __name__ == "__main__":
    main()
    exit(0)
    