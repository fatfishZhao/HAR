# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import PIL.Image as Image
from dataset.data_aug import *
import cv2
import numpy as np

class dataset(data.Dataset):
    def __init__(self, dataroot, anno_pd, transforms=None, mode = 'train'):
        self.root_path = dataroot
        if mode == 'train':
            x_train = np.load(os.path.join(dataroot, 'x_train.npy'))
            # Nomalize
            y_train = np.load(os.path.join(dataroot, 'y_train.npy'))
            y_train = [0 if each_y==1 else 0 for each_y in y_train]
            self.data = x_train; self.labels = y_train
        if mode =='val':
            x_val = np.load(os.path.join(dataroot, 'x_test.npy'))
            y_val = np.load(os.path.join(dataroot, 'y_test.npy'))
            y_val = [0 if each_y == 1 else 0 for each_y in y_val]
            self.data = x_val;  self.labels = y_val

        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        each_data = self.data[item]



        if self.transforms is not None:
            each_data = self.transforms(each_data)
        label = self.labels[item]

        return torch.from_numpy(each_data).float(), label

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
def collate_fn(batch):
    imgs = []
    label = []

    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), \
           label