import pickle
import random
import os
import torch

import src.dataman.vocab as vocab
import src.constant as constant

class TextDataManager:
    def __init__(self, args):
        self.config = args
        self.max_len = args.max_len
        self.vocab = vocab.Vocab(logdir=constant.DATA_DIR)
        self.vocab.load_glove(path=constant.GLOVE_DIR, d_embed=args.glove_embedding_size)
        self.vocab.load_medw2v(path=constant.DATA_DIR)

        with open(constant.TRAIN_IDS, "r") as fd:
            lines = fd.readlines()
            dbids = [line.strip('\n') for line in lines]
            self.train_dbids = dbids[0:args.train_num_drugs]
            self.dev_dbids = dbids[args.train_num_drugs:]
        with open(constant.TEST_IDS, "r") as fd:
            lines = fd.readlines()
            self.test_dbids = [line.strip('\n') for line in lines]

        with open(constant.TRAIN_LABELS, "rb") as fd:
            self.train_label_dict = pickle.load(fd)
        with open(constant.TEST_LABELS, "rb") as fd:
            self.test_label_dict = pickle.load(fd)

        self.features = {tag: [] for tag in constant.feature_tags}
        for tag in constant.feature_tags:
            with open(os.path.join(constant.DATA_DIR, 'text_' + 'dbid2' + tag + '_tok.pkl'), 'rb') as fd:
                self.features[tag] = pickle.load(fd)

        self.train_sents_tensor = {}
        self.train_sents_len = {}
        self.dev_sents_tensor = {}
        self.dev_sents_len = {}
        self.test_sents_tensor = {}
        self.test_sents_len = {}
        for tag in constant.feature_tags:
            train_sents = [self.features[tag][dbid] for dbid in self.train_dbids]
            self.train_sents_tensor[tag], self.train_sents_len[tag] = self.get_sents_tensor(train_sents)
            dev_sents = [self.features[tag][dbid] for dbid in self.dev_dbids]
            self.dev_sents_tensor[tag], self.dev_sents_len[tag] = self.get_sents_tensor(dev_sents)
            test_sents = [self.features[tag][dbid] for dbid in self.test_dbids]
            self.test_sents_tensor[tag], self.test_sents_len[tag] = self.get_sents_tensor(test_sents)
        self.train_labels = torch.Tensor([self.train_label_dict[dbid] for dbid in self.train_dbids])
        self.dev_labels = torch.Tensor([self.train_label_dict[dbid] for dbid in self.dev_dbids])
        self.test_labels = torch.Tensor([self.test_label_dict[dbid] for dbid in self.test_dbids])


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

    def sample_batch(self, batch_size, sents_tensor, sents_len, embed1, embed2, use_cuda):
        data_size = len(sents_tensor['description'])
        perm = torch.randperm(data_size)
        idx = perm[:batch_size]
        batch_des_sents_tensor = sents_tensor['description'][idx]
        batch_des_sents_len = sents_len['description'][idx]
        batch_ind_sents_tensor = sents_tensor['indication'][idx]
        batch_ind_sents_len = sents_len['indication'][idx]
        batch_act_sents_tensor = sents_tensor['mechanism-of-action'][idx]
        batch_act_sents_len = sents_len['mechanism-of-action'][idx]
        batch_target = self.train_labels[idx]

        des_packed_tensor, des_idx_unsort = self.vocab.get_packedseq_from_sent_batch(
            seq_tensor=batch_des_sents_tensor,
            seq_lengths=batch_des_sents_len,
            embed1=embed1,
            embed2=embed2,
            use_cuda=use_cuda,
        )
        ind_packed_tensor, ind_idx_unsort = self.vocab.get_packedseq_from_sent_batch(
            seq_tensor=batch_ind_sents_tensor,
            seq_lengths=batch_ind_sents_len,
            embed1=embed1,
            embed2=embed2,
            use_cuda=use_cuda,
        )
        act_packed_tensor, act_idx_unsort = self.vocab.get_packedseq_from_sent_batch(
            seq_tensor=batch_act_sents_tensor,
            seq_lengths=batch_act_sents_len,
            embed1=embed1,
            embed2=embed2,
            use_cuda=use_cuda,
        )

        return (des_packed_tensor,
                des_idx_unsort,
                ind_packed_tensor,
                ind_idx_unsort,
                act_packed_tensor,
                act_idx_unsort,
                batch_target)

    def sample_train_batch(self, batch_size, embed1, embed2, use_cuda):
        return self.sample_batch(batch_size, self.train_sents_tensor, self.train_sents_len,
                                 embed1=embed1, embed2=embed2, use_cuda=use_cuda)

    def sample_dev_batch(self, batch_size, embed1, embed2, use_cuda):
        return self.sample_batch(batch_size, self.dev_sents_tensor, self.dev_sents_len,
                                 embed1=embed1, embed2=embed2, use_cuda=use_cuda)

    def sample_test_batch(self, batch_size, embed1, embed2, use_cuda):
        return self.sample_batch(batch_size, self.test_sents_tensor, self.test_sents_len,
                                 embed1=embed1, embed2=embed2, use_cuda=use_cuda)