import numpy as np
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import logging
import datetime
import threading
import queue

motion_dict = {'jog': 0,
               'skip': 1,
               'stay': 2,
               'stdown': 3,
               'stup': 4,
               'walk': 5}
def dt():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def get_train_val_dataset(personid=None, transforms = None, window_size = 256,overlap=0.5):
    assert personid is not None
    data_root = '/data3/zyx/project/HAR/data'
    meta_df = pd.DataFrame(columns=['motion', 'personid', 'metaid'])
    motions = []; personids = []; metaids = []
    for key in motion_dict.keys():
        tmp_personmap = np.load(os.path.join(data_root, key+'user1.csvpersonmap.npy')).item()
        for each_person in tmp_personmap.keys():
            if each_person==personid:
                continue
            motions += [key]*len(tmp_personmap[each_person])
            personids += [each_person]*len(tmp_personmap[each_person])
            metaids += tmp_personmap[each_person]
    meta_df['motion'] = motions
    meta_df['personid'] = personids
    meta_df['metaid'] = metaids
    train_meta, val_meta = train_test_split(meta_df, test_size=0.02, random_state=43,
                                    stratify=meta_df['motion'])
    train_dataset = dataset(dataroot=data_root,anno_pd=train_meta, transforms=transforms['train'], window_size=window_size,overlap=overlap)
    val_dataset = dataset(data_root, val_meta,transforms=transforms['val'], window_size=window_size)
    logging.info('%s'%(dt())+'train data size,'+str(len(train_dataset)))
    logging.info('%s'%(dt())+'val data size,' + str(len(val_dataset)))
    return train_dataset, val_dataset
def get_window_data(match_data, window_size, overlap):
    np_data = match_data.iloc[:,2:8].values
    start_indexs = np.arange(0, np_data.shape[0]-window_size, int(window_size*(1-overlap)))
    window_data = []
    for start_index in start_indexs:
        window_data += [np_data[start_index:start_index+window_size,:]]
    return window_data
def get_one_motion_data(dataroot,motion,data_label_q, metas,window_size, overlap):
    raw_data = pd.read_csv(os.path.join(dataroot, motion+'user1.csv'))
    logging.info('%s'%(dt())+'loaded data from '+motion)
    for meta in metas:
        match_data = raw_data[raw_data['metaid']==meta]
        tmp_window_data = get_window_data(match_data, window_size, overlap)
        data_label_q.put((tmp_window_data, [motion_dict[motion]]*len(tmp_window_data)))


class dataset(data.Dataset):
    def __init__(self, dataroot, anno_pd, transforms=None, window_size = 256, overlap = 0.5):
        # self.root_path = imgroot
        all_motions = set(anno_pd['motion'].values)
        self.data = []
        self.labels = []
        data_label_queue = queue.Queue(maxsize=0)
        for motion in all_motions:
            # raw_data = pd.read_csv(os.path.join(dataroot,motion+'user1.csv'))
            # logging.info('%s'%(dt())+'loaded data from '+motion)
            tmp_meta = anno_pd[anno_pd['motion']==motion]
            t = threading.Thread(target=get_one_motion_data, args=(dataroot, motion, data_label_queue, tmp_meta['metaid'].values, window_size, overlap))
            t.setDaemon(True)
            t.start()
            # for meta in tmp_meta['metaid'].values:
                # match_data = raw_data[raw_data['metaid']==meta]
                # tmp_window_data = get_window_data(match_data,window_size, overlap)

        # self.data += tmp_window_data
        # self.labels += [motion_dict[motion]]*len(tmp_window_data)
                # print(match_data)
        t.join()
        print(data_label_queue)
        print('%s read over'%(dt()))
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        data = self.data[item]
        label = self.labels[item]

        if self.transforms is not None:
            img = self.transforms(data)

        return torch.from_numpy(img).float(), label


def collate_fn(batch):
    imgs = []
    label = []

    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), \
           label
if __name__=='__main__':
    personmap = np.load('/data/yxzhao_data/project/finetuneAR/data/joguser1.csvpersonmap.npy').item()
    personids = list(personmap.keys())
    get_train_val_dataset(personids[0])