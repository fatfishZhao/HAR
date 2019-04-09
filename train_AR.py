#coding=utf-8
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.AR_dataset import collate_fn, dataset, get_train_val_dataset
import torch
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from torchvision.models import resnet50
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.train_util import train, trainlog
from  torch.nn import CrossEntropyLoss
import logging
from dataset.data_aug import *
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''

'''
save_dir = '/data/yxzhao_data/project/finetuneAR/trained_model/resnet50_2_256'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logfile = '%s/trainlog.log'%save_dir
trainlog(logfile)

data_root = '/data/yxzhao_data/project/finetuneAR/data'
personmap = np.load(os.path.join(data_root, 'jog'+'user1.csvpersonmap.npy')).item()
test_person = list(personmap.keys())[0]

'''数据扩增'''
data_transforms = {
    'train': Compose([
        sensor_to3d(mode='train'),
        ExpandBorder(size=(256,256),resize=True,value=0),
        sensor_transpose(mode='train')
    ]),
    'val': Compose([
        sensor_to3d(mode='val'),
        ExpandBorder(size=(256,256),resize=True,value=0),
        sensor_transpose(mode='val')
    ]),
}

data_set = {}
data_set['train'], data_set['val'] = get_train_val_dataset(test_person, transforms=data_transforms)
dataloader = {}
dataloader['train']=torch.utils.data.DataLoader(data_set['train'], batch_size=16,
                                               shuffle=True, num_workers=4,collate_fn=collate_fn)
dataloader['val']=torch.utils.data.DataLoader(data_set['val'], batch_size=16,
                                               shuffle=True, num_workers=4,collate_fn=collate_fn)
'''model'''
model =resnet50(pretrained=True)
model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
model.fc = torch.nn.Linear(model.fc.in_features,6)

base_lr =0.001
resume =None
if resume:
    logging.info('resuming finetune from %s'%resume)
    model.load_state_dict(torch.load(resume))
model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-5)
criterion = CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

train(model,
      epoch_num=50,
      start_epoch=0,
      optimizer=optimizer,
      criterion=criterion,
      exp_lr_scheduler=exp_lr_scheduler,
      data_set=data_set,
      data_loader=dataloader,
      save_dir=save_dir,
      print_inter=50,
      val_inter=400)