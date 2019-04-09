import os
import numpy as np
import pandas as pd
from dataset.AR_dataset import get_test_dataset, collate_fn
import torch
from torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision.models import resnet50, resnet18
from torch.autograd import Variable
from math import ceil
from dataset.data_aug import *
from sklearn.metrics import confusion_matrix, classification_report
from dataset.AR_dataset import motion_dict

data_root = '/data3/zyx/project/HAR/data'
personmap = np.load(os.path.join(data_root, 'jog' + 'user1.csvpersonmap.npy')).item()
test_person = list(personmap.keys())[0]
window_size = 512
overlap = 0.5
test_transforms = {
    'test':Compose([
        sensor_to3d(mode='train'),
        ExpandBorder(size=(window_size, window_size), resize=True, value=0),
        sensor_transpose(mode='test')
    ])}

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
data_set = {}
data_set['test'] = get_test_dataset(test_person, transforms=test_transforms,
                                                           window_size=window_size, overlap=overlap)

data_loader = {}
data_loader['test'] = torchdata.DataLoader(data_set['test'], batch_size=4, num_workers=4,
                                           shuffle=False, pin_memory=True, collate_fn=collate_fn)

model_name = 'resnet50_1212_2'
resume = '/data3/zyx/project/HAR/trained_model/resnet50_1212_2/weights-1-5578-[0.9033].pth'

model =resnet50(pretrained=True)
model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
model.fc = torch.nn.Linear(model.fc.in_features,6)

print('resuming finetune from %s'%resume)
model.load_state_dict(torch.load(resume))
model = model.cuda()
model.eval()

criterion = CrossEntropyLoss()

if not os.path.exists('./Baidu/csv'):
    os.makedirs('./Baidu/csv')

test_size = ceil(len(data_set['test']) / data_loader['test'].batch_size)
test_preds = np.zeros((len(data_set['test'])), dtype=np.float32)
true_label = np.zeros((len(data_set['test'])), dtype=np.int)
idx = 0
test_loss = 0
test_corrects = 0
for batch_cnt_test, data_test in enumerate(data_loader['test']):
    # print data
    print("{0}/{1}".format(batch_cnt_test, int(test_size)))
    inputs, labels = data_test
    inputs = Variable(inputs.cuda())
    labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
    # forward
    outputs = model(inputs)

    # statistics
    if isinstance(outputs, list):
        loss = criterion(outputs[0], labels)
        loss += criterion(outputs[1], labels)
        outputs = (outputs[0]+outputs[1])/2
    else:
        loss = criterion(outputs, labels)
    _, preds = torch.max(outputs, 1)

    test_loss += loss.item()
    batch_corrects = torch.sum((preds == labels)).item()
    test_corrects += batch_corrects
    test_preds[idx:(idx + labels.size(0))] = preds
    true_label[idx:(idx + labels.size(0))] = labels.data.cpu().numpy()
    # statistics
    idx += labels.size(0)
test_loss = test_loss / test_size
test_acc = 1.0 * test_corrects / len(data_set['test'])
print('test-loss: %.4f ||test-acc@1: %.4f'
      % (test_loss, test_acc))
print('---------matrix---------')
conf_matrix = confusion_matrix(true_label, test_preds)
print(conf_matrix)
print('--------report----------')
print(classification_report(true_label, test_preds, target_names=list(motion_dict.keys())))

