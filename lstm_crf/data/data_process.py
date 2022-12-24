from glob import glob
import os
import random
import pandas as pd

import config
from config import *

"""
转换brat格式
进行1对1处理
example:
brat原格式.ana{
T1 人名 11 13 马云
T2 地点 16 18 杭州
R1 来自 Arg1:T1 Arg2:T2
T3 人名 23 25 马云
T4 时间 0 10 1964年9月10日
T5 人名 61 63 父亲
T6 人名 81 83 马云
T7 地点 69 71 江南
T8 人名 99 101 金庸
T9 人名 129 131 马云
}
out格式:
1，时间
9，时间
6，时间
3，时间
年，时间
...
马，人名
云，人民
在，0

"""


def get_annotation(ann_path):
    with open(ann_path, encoding='utf-8') as file:
        ans = {}
        for line in file.readlines():
            temp = line.split(' ')
            name = temp[1]
            # 人名，时间
            start = int(temp[2])
            end = int(temp[-2])
            if (end - start > 50):
                continue
            ans[start] = 'b-' + name
            for i in range(start + 1, end):
                ans[i] = 'i-' + name
        return ans


def get_text(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def pire_ann():
    for text_path in glob(ORIGIN_PATH + '*.text'):
        ann_path = text_path[:-4] + 'ann'
        anns = get_annotation(ann_path)
        text = get_text(text_path)
        # 标注
        df = pd.DataFrame({'word': list(text), 'label': ['0'] * len(text)})
        df.loc[anns.keys(), 'label'] = list(anns.values())
        # 导出
        file_name = os.path.split(text_path)[1]
        df.to_csv(ANN_PATH + file_name)


def merge_text(text_path_list, save_path):
    # text_path为可迭代对象，例如glob()产生的list
    with open(save_path, encoding='utf-8') as f1:
        for text_path in text_path_list:
            with open(text_path, encoding='utf-8') as f2:
                temp = f2.read()
                f1.write(temp)


# 默认测试占0.2
def split_sample(test_size=0.2):
    f = glob(ORIGIN_PATH + '*.text')
    random.seed(0)
    random.shuffle(f)
    n = int(len(f) * test_size)
    # 分test和train
    test_file = f[:n]
    train_file = f[n:]
    # 输出为test_text和train_text
    merge_text(test_file, TEST_SAMPLE_PATH)
    merge_text(train_file, TRAIN_SAMPLE_PATH)


# 生成词汇表
def generate_vocab():
    df = pd.read_csv(TRAIN_SAMPLE_PATH, usecols=[0], names=['word'])
    vocab_list = [WORD_PAD, WORD_UNK] + df['word'].value_counts().keys().to_list()
    vocab_list = vocab_list[:VOCAB_SIZE]
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab = pd.DataFrame(list(vocab_dict.items()))
    vocab.to_csv(VOCAB_PATH, header=None, index=None)


# 生成label表
def generate_label():
    df = pd.read_csv(TRAIN_SAMPLE_PATH, usecols=[1], names=['label'])
    label_list = df['label'].value_counts().keys().to_list()
    label_dict = {v: k for k, v in enumerate(label_list)}
    label = pd.DataFrame(list(label_dict.values()))
    label.to_csv(LABEL_PATH, header=None, index=None)


if __name__ == '__main__':
    anss = get_annotation('./input/tset1.ann')
    print(anss)
    print(get_text('./input/tset1.ann'))
