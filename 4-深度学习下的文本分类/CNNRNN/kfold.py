#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time:2022-04-17 15:48
@Author:Veigar
@File: kfold.py
@Github:https://github.com/veigaran
"""
""""
在run.py上改写，增加一个k-fold的功能，比较简单，也可以用shell脚本的方式
"""
import os
import time
import torch
import numpy as np
import pandas as pd
from train_eval import train, test, init_network
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='TextCNN',
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


def process_one_fold(dataset, embedding):
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)

    # 单独评测，则把上方train部分注释掉
    # test(config, model, test_iter)


def batch(folder, embed):
    filelist = os.listdir(folder)
    for fold in filelist:
        dataset = os.path.join(folder, fold)
        process_one_fold(dataset, embed)



def txt2npz(txt,npz):
    embeddings_index = {}
    with open(txt, 'r', encoding='utf-8')as f:
        for line in f.readlines()[1:]:
            values = line.split()
            word = values[0]
            embeddings_index[word] = np.asarray(values[1:], dtype='float32')
    my_df = pd.DataFrame.from_dict(embeddings_index).T
    print(my_df.shape)
    np.save(npz, my_df)


if __name__ == '__main__':
    folder = 'corpus'
    # embed = 'embedding_SougouNews.npz'
    embed = 'baidu.npy'
    batch(folder, embed)

    # txt2npz('pretrain/embedding_baidu.csv','baidu.npy')




