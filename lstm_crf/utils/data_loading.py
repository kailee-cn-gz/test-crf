import torch
from torch.utils import data
from config import *
import pandas as pd


def get_vocab():
    df = pd.read_csv(VOCAB_PATH, names=['word', 'id'])
    l = list(df['word'])
    d = dict(df.values)
    return l, d


def get_label():
    df = pd.read_csv(LABEL_PATH, names=['label', 'id'])
    l = list(df['label'])
    d = dict(df.values)
    return l, d


class dataest(data.Dataset):
    def __init__(self, type='train', base_l=50):
        super(dataest, self).__init__()
        self.base_l = base_l
        if (type == 'train'):
            path = TRAIN_SAMPLE_PATH
        else:
            path = TEST_SAMPLE_PATH
        self.df = pd.read_csv(path, names=['word', 'label'])
        _, self.word2id = get_vocab()
        _, self.label2id = get_label()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
