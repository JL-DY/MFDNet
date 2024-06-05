import os
import time
import argparse
import torch
import torch.nn as nn
from utils.utils import jl_DatasetFromMat, cur_timestamp_str
from torch.utils.data import DataLoader
from model.MFDNet import MFDNet

parser = argparse.ArgumentParser(description="PyTorch MFDNet")

# paramters for training
parser.add_argument("--batchSize", type=int, default=40, help="Training batch size")
parser.add_argument("--epoch", type=int, default=100000, help="Number of epoch to train for")
parser.add_argument("--lr", type=float, default=0.0004, help="Learning Rate. Default=0.1")
parser.add_argument('--dataset', default='/home/jl/Project/DnCNN-unofficial/Train400', type=str, help='path to general model')
parser.add_argument("--sigma", default=8, type=int, help="add gaussian white noise level")
parser.add_argument("--pretrain", default=None, help="path of pretrained model")
parser.add_argument("--checkpoint", type=str, default="./weights", help="save model path")
parser.add_argument("--img_size", type=tuple, default=(360, 360), help="training image size(weight, height), It must be divisible by both 4 and 6")
parser.add_argument("--per_epoch_save", type=int, default=1000, help="Save the model weight every number of epochs")
parser.add_argument("--test_input", type=str, default="./input_imgs", help="test image input folder")
parser.add_argument("--test_output", type=str, default="./output_imgs/noise11", help="test image save folder")
parser.add_argument("--test_model", type=str, default="./weights/mfdnet-m3k3c16-1209-1738-noise11/best_model.pt")

# paramters for MFDNet
parser.add_argument("--M_MFDB", type=int, default=3, help="The number of MFDB")
parser.add_argument("--K_RepConv", type=int, default=3, help="The number of RepConv in each MFDB")
parser.add_argument("--c_channel", type=int, default=16, help="The number of channels in conv3*3") # 论文中应该是三通道，所以做两次haar变换c才会是48

## hardware specification
parser.add_argument("--cuda", default=1, type=int, help="cuda:0, 1...; cpu:-1")
parser.add_argument("--threads", default=1, type=int, help="number of data load for training")

opt = parser.parse_args()

def main():
    # set hardware specification
    if opt.cuda >=0 and torch.cuda.is_available():
        print("use cuda & cudnn for acceleration!")
        print("the gpu id is:{}".format(opt.cuda))
        device = torch.device('cuda:{}'.format(opt.cuda))
    else:
        print("use cpu for training or testing")
        device = torch.device('cpu')
    torch.set_num_threads(opt.threads)

    # set train dataloader
    train_data_set = jl_DatasetFromMat(opt.dataset, opt.sigma, opt.img_size)
    train_dataloader = DataLoader(train_data_set, batch_size=opt.batchSize, shuffle=True, num_workers=opt.threads, drop_last=True)

    # set model、optimizer and loss
    model = MFDNet(opt.M_MFDB, opt.K_RepConv, opt.c_channel).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(opt.epoch/10), gamma=0.5)
    loss_func = nn.MSELoss(size_average=False)
    if opt.pretrain is not None:
        print("load pretrainedd model: {}!".format(opt.pretrain))
        model.load_state_dict(torch.load(opt.pretrain))
    else:
        print("train the model from scratch!")

    # auto-generate the output logname
    timestamp = cur_timestamp_str()
    experiment_folder = "mfdnet-m{}k{}c{}-{}".format(opt.M_MFDB, opt.K_RepConv, opt.c_channel, timestamp)
    experiment_path = os.path.join(opt.checkpoint, experiment_folder)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    # start training
    best_epoch = 0
    best_loss = 1e6
    total_steps = len(train_dataloader.dataset)//opt.batchSize
    for epoch in range(opt.epoch):
        epoch_loss = 0.0
        timer_start = time.time()
        model = model.train()
        for iter, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            noise_img, clear_img = batch
            noise_img, clear_img = noise_img.to(device), clear_img.to(device)

            out_img = model(noise_img)

            loss = loss_func(out_img, clear_img)/(opt.batchSize * 2)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss)
            avg_loss = epoch_loss / (iter+1)
        # print training log
        duration = time.time() - timer_start
        print("--------------------------------")
        print("Epoch:{}, {}/{}, loss: {:.4f}, cur_lr:{}, time: {:.3f}".format(epoch, iter, total_steps, avg_loss, optimizer.state_dict()['param_groups'][0]['lr'], duration))
        if(avg_loss < best_loss):
            best_loss = avg_loss
            best_epoch = epoch
            save_model_path = os.path.join(experiment_path, "best_model.pt")
            torch.save(model.state_dict(), save_model_path)
        print("best epoch:{}, best loss:{}".format(best_epoch, best_loss))
        scheduler.step()
        # save model paramters
        # if((epoch+1) % opt.per_epoch_save == 0):
        #     save_model_path = os.path.join(experiment_path, "model_{}.pt".format(epoch))
        #     torch.save(model.state_dict(), save_model_path)

if __name__ == "__main__":
    main()
    exit(0)
