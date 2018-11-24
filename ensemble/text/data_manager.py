import pickle
import random
import os
import torch

from ensemble.data_manager import DataManager

import src.dataman.vocab as vocab
import ensemble.constant as constant


class TextDataManager(DataManager):
    def __init__(self, train_dbids, dev_dbids, test_dbids, use_cuda, max_len, glove_embedding_size):
        super(TextDataManager, self).__init__(train_dbids, dev_dbids, test_dbids, use_cuda)
        print('Initializing TextDataManager...')
        self.max_len = max_len
        self.vocab = vocab.Vocab(logdir=constant.DATA_DIR)
        self.vocab.load_glove(path=constant.GLOVE_DIR, d_embed=glove_embedding_size)
        self.vocab.load_medw2v(path=constant.DATA_DIR)

        self.features = {tag: [] for tag in constant.TEXT_FEATURE_TAGS}
        for tag in constant.TEXT_FEATURE_TAGS:
            with open(os.path.join(constant.DATA_DIR, 'text_' + 'dbid2' + tag + '_tok.pkl'), 'rb') as fd:
                self.features[tag] = pickle.load(fd)

        self.train_sents_tensor = {}
        self.train_sents_len = {}
        self.dev_sents_tensor = {}
        self.dev_sents_len = {}
        self.test_sents_tensor = {}
        self.test_sents_len = {}
        for tag in constant.TEXT_FEATURE_TAGS:
            train_sents = [self.features[tag][dbid] for dbid in self.train_dbids]
            self.train_sents_tensor[tag], self.train_sents_len[tag] = self.get_sents_tensor(train_sents)
            dev_sents = [self.features[tag][dbid] for dbid in self.dev_dbids]
            self.dev_sents_tensor[tag], self.dev_sents_len[tag] = self.get_sents_tensor(dev_sents)
            test_sents = [self.features[tag][dbid] for dbid in self.test_dbids]
            self.test_sents_tensor[tag], self.test_sents_len[tag] = self.get_sents_tensor(test_sents)

        self.embed1 = None
        self.embed2 = None
        self.train_dbid_to_idx = {
            dbid: i for i, dbid in enumerate(self.train_dbids)
        }
        self.dev_dbid_to_idx = {
            dbid: i for i, dbid in enumerate(self.dev_dbids)
        }
        self.test_dbid_to_idx = {
            dbid: i for i, dbid in enumerate(self.test_dbids)
        }
        print('TextDataManager initialized.')

    def connect_to_submodule(self, submodule):
        self.embed1 = submodule.glove_embed
        self.embed2 = submodule.other_embed

    def get_sents_tensor(self, sents_num):
        """ sents_num: list of lists of word ids """
        data_size = len(sents_num)
        r_tensor = torch.LongTensor(data_size, self.max_len)
        r_tensor.fill_(vocab.PAD_token)
        slen_tensor = torch.IntTensor(data_size,)

        b = 0
        for sent_wordids in sents_num:
            slen = min(len(sent_wordids), self.max_len)
            if slen == 0:
                print(sent_wordids)
            slen_tensor[b] = slen
            for w in range(slen):
                wordid = sent_wordids[w]
                r_tensor[b, slen-1-w] = wordid  # fill in reverse
            b += 1
        return r_tensor, slen_tensor

    def sample_batch(self, dbids, dbid_to_idx, sents_tensor, sents_len):
        idx = [dbid_to_idx[dbid] for dbid in dbids]
        batch_des_sents_tensor = sents_tensor['description'][idx]
        batch_des_sents_len = sents_len['description'][idx]
        batch_ind_sents_tensor = sents_tensor['indication'][idx]
        batch_ind_sents_len = sents_len['indication'][idx]
        batch_act_sents_tensor = sents_tensor['mechanism-of-action'][idx]
        batch_act_sents_len = sents_len['mechanism-of-action'][idx]

        des_packed_tensor, des_idx_unsort = self.vocab.get_packedseq_from_sent_batch(
            seq_tensor=batch_des_sents_tensor,
            seq_lengths=batch_des_sents_len,
            embed1=self.embed1,
            embed2=self.embed2,
            use_cuda=self.use_cuda,
        )
        ind_packed_tensor, ind_idx_unsort = self.vocab.get_packedseq_from_sent_batch(
            seq_tensor=batch_ind_sents_tensor,
            seq_lengths=batch_ind_sents_len,
            embed1=self.embed1,
            embed2=self.embed2,
            use_cuda=self.use_cuda,
        )
        act_packed_tensor, act_idx_unsort = self.vocab.get_packedseq_from_sent_batch(
            seq_tensor=batch_act_sents_tensor,
            seq_lengths=batch_act_sents_len,
            embed1=self.embed1,
            embed2=self.embed2,
            use_cuda=self.use_cuda,
        )

        return (des_packed_tensor,
                des_idx_unsort,
                ind_packed_tensor,
                ind_idx_unsort,
                act_packed_tensor,
                act_idx_unsort)

    def sample_train_batch(self, dbids):
        return self.sample_batch(dbids, self.train_dbid_to_idx, self.train_sents_tensor, self.train_sents_len)

    def sample_dev_batch(self, dbids):
        return self.sample_batch(dbids, self.dev_dbid_to_idx, self.dev_sents_tensor, self.dev_sents_len)

    def sample_test_batch(self, dbids):
        return self.sample_batch(dbids, self.test_dbid_to_idx, self.test_sents_tensor, self.test_sents_len)
