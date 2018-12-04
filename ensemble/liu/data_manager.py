from ensemble.data_manager import DataManager
from ensemble import constant
import os
import csv
import pickle
from ensemble.protein import vocab
from src.utils import l2_normalize
import torch
import numpy as np
from torch import nn


class LiuDataManager(DataManager):
    def __init__(self, train_dbids, dev_dbids, test_dbids, use_cuda):
        super(LiuDataManager, self).__init__(train_dbids, dev_dbids, test_dbids, use_cuda)
        print('Initializing LiuDataManager...')
        with open('/home/ryanlh/train_dbids_munoz.txt', 'rt') as infile:
            reader = csv.reader(infile)
            train_dbid_order = {
                row[0]: i for i, row in enumerate(reader)
            }
        with open('/home/ryanlh/test_dbids_munoz.txt', 'rt') as infile:
            reader = csv.reader(infile)
            test_dbid_order = {
                row[0]: i for i, row in enumerate(reader)
            }

        self.train_features = np.zeros((len(train_dbids), 0))
        self.dev_features = np.zeros((len(dev_dbids), 0))
        self.test_features = np.zeros((len(test_dbids), 0))
        for fname in os.listdir('/data/munoz2017_sider4_data/original'):
            if fname.startswith('test') or 'sideEffect' in fname:
                continue
            d_train = pickle.load(open('/data/munoz2017_sider4_data/original/{}'.format(fname), 'rb'))
            d_test = pickle.load(
                open('/data/munoz2017_sider4_data/original/{}'.format(fname.replace('train', 'test')), 'rb'))
            train_arr = []
            dev_arr = []
            test_arr = []
            for train_dbid in train_dbids:
                train_arr.append(d_train[train_dbid_order[train_dbid], :])
            for dev_dbid in dev_dbids:
                dev_arr.append(d_train[train_dbid_order[dev_dbid], :])
            for test_dbid in test_dbids:
                test_arr.append(d_test[test_dbid_order[test_dbid], :])
            self.train_features = np.hstack([self.train_features, train_arr])
            self.dev_features = np.hstack([self.dev_features, dev_arr])
            self.test_features = np.hstack([self.test_features, test_arr])
        self.num_terms = self.train_features.shape[1]
        self.train_features = torch.FloatTensor(self.train_features)
        self.dev_features = torch.FloatTensor(self.dev_features)
        self.test_features = torch.FloatTensor(self.dev_features)
        if use_cuda:
            self.train_features = self.train_features.cuda()
            self.dev_features = self.dev_features.cuda()
            self.test_features = self.test_features.cuda()

        print('LiuDataManager initialized.')

    def connect_to_submodule(self, submodule):
        pass

    def sample_batch(self, dbids, dbid_to_idx, feature_tensor):
        idx = [dbid_to_idx[dbid] for dbid in dbids]

        return torch.FloatTensor(feature_tensor[idx, :])

    def sample_train_batch(self, dbids):
        return self.sample_batch(dbids, self.train_dbid_to_idx, self.train_features)

    def sample_dev_batch(self, dbids):
        return self.sample_batch(dbids, self.dev_dbid_to_idx, self.dev_features)

    def sample_test_batch(self, dbids):
        return self.sample_batch(dbids, self.test_dbid_to_idx, self.test_features)
