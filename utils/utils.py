import torch
import torch.utils.data as data
import os
import cv2 as cv
import numpy as np
import datetime


class jl_DatasetFromMat(data.Dataset):
    def __init__(self, file_path, sigma, img_size):
        super(jl_DatasetFromMat, self).__init__()
        self.sigma = sigma
        self.file_path = file_path
        self.img_size = img_size
        self.img_name = os.listdir(file_path)
        self.rnd_aug = np.random.randint(8, size=len(self.img_name))

    def __len__(self):
        # return self.data.Fileshape[0]
        return len(self.img_name)
    
    def __getitem__(self, index):
        img = cv.imread(os.path.join(self.file_path, self.img_name[index]))
        img = cv.resize(img, self.img_size)
        img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        img_y = np.expand_dims(np.array(img)[:, :, 0], -1)

        img_y_aug = data_aug(img_y, self.rnd_aug[index])
        img_y_aug = np.transpose(img_y_aug, (2, 0, 1)) / 255
        label = np.random.normal(0, self.sigma, size=np.shape(img_y_aug)) / 255
        input = img_y_aug + label
        return torch.from_numpy(input).float(), torch.from_numpy(img_y_aug).float()


# 数据增强部分
def data_aug(image, mode=0):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        return np.flipud(np.rot90(image))
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        return np.flipud(np.rot90(image, k=2))
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        return np.flipud(np.rot90(image, k=3))
    

# 获取时间
def cur_timestamp_str():
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)

    content = "{}-{}{}-{}{}".format(year, month, day, hour, minute)
    return content