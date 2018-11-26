from ensemble.data_manager import DataManager
from ensemble import constant
import os
import pickle
from ensemble.protein import vocab
from src.utils import l2_normalize
import torch
from torch import nn


class GoDataManager(DataManager):
    def __init__(self, train_dbids, dev_dbids, test_dbids, use_cuda):
        super(GoDataManager, self).__init__(train_dbids, dev_dbids, test_dbids, use_cuda)
        print('Initializing GoDataManager...')
        with open('/data/data_targetProteins/goterms_to_idx.pkl', 'rb') as infile:
            go_to_idx = pickle.load(infile)
        with open('/data/data_targetProteins/train_target_go.pkl', 'rb') as infile:
            dbid_to_gos = pickle.load(infile)
        with open('/data/data_targetProteins/test_target_go.pkl', 'rb') as infile:
            dbid_to_gos.update(pickle.load(infile))
        self.num_terms = len(go_to_idx)
        self.train_gos = [set(go_to_idx[go] for gos in dbid_to_gos[dbid] for go in gos) for dbid in self.train_dbids]
        self.dev_gos = [set(go_to_idx[go] for gos in dbid_to_gos[dbid] for go in gos) for dbid in self.dev_dbids]
        self.test_gos = [set(go_to_idx[go] for gos in dbid_to_gos[dbid] for go in gos) for dbid in self.test_dbids]

        print('GoDataManager initialized.')

    def connect_to_submodule(self, submodule):
        pass

    def sample_batch(self, dbids, dbid_to_idx, go_list):
        idx = [dbid_to_idx[dbid] for dbid in dbids]

        if self.use_cuda:
            return torch.cuda.FloatTensor([[1 if g in go_list[i] else 0 for g in range(self.num_terms)] for i in idx])
        else:
            return torch.FloatTensor([[1 if g in go_list[i] else 0 for g in range(self.num_terms)] for i in idx])

    def sample_train_batch(self, dbids):
        return self.sample_batch(dbids, self.train_dbid_to_idx, self.train_gos)

    def sample_dev_batch(self, dbids):
        return self.sample_batch(dbids, self.dev_dbid_to_idx, self.dev_gos)

    def sample_test_batch(self, dbids):
        return self.sample_batch(dbids, self.test_dbid_to_idx, self.test_gos)
