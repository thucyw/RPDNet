import os
import scipy.io as scio
import torch
from torch.utils.data import Dataset
import numpy as np

def load_data(datapath, labelpath = None):
    
    norm = lambda x: (x - x.min())/(x.max() - x.min())

    files = os.listdir(datapath)
    files.sort()
    data = [norm(scio.loadmat(os.path.join(datapath, i))['rdm'][np.newaxis]) for i in files]

    if not labelpath:
        return data
    
    files = os.listdir(labelpath)
    files.sort()
    label = [scio.loadmat(os.path.join(labelpath, i))['label'][np.newaxis] for i in files]

    assert(len(data)==len(label))

    return data, label


def init_dataset(config):

    if config.act == "train" or config.act == "eval":

        D, L = [], []
        for i in config.data.train:
            data, label = load_data("data/{}/data/".format(i), "data/{}/label/".format(i))
            D += data
            L += label
        train_dataset = myDataset(D, L)
        
        D, L = [], []
        for i in config.data.test:
            data, label = load_data("data/{}/data/".format(i), "data/{}/label/".format(i))
            D += data
            L += label
        test_dataset = myDataset(D, L)

        print("Using {} to train, {} to test".format(config.data.train, config.data.test))
        print("Train data - {}, Test data - {}".format(train_dataset.len, test_dataset.len))

        return train_dataset, test_dataset

    if config.act == "transform":

        transform_datasets = []
        for i in config.data.transform:
            data = load_data("data/{}/data/".format(i))
            transform_datasets += myDataset([data], [])

        return transform_datasets


class myDataset(Dataset):
    def __init__(self, data, label = None):

        self.data = data
        self.label = label
        self.len = len(data)

    def __getitem__(self, index):
        
        if self.label:
            return self.data[index], self.label[index]
        return self.data[index]

    def __len__(self):

        return self.len
