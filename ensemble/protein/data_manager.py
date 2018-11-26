from ensemble.data_manager import DataManager
from ensemble import constant
import os
import pickle
from ensemble.protein import vocab
from src.utils import l2_normalize
import torch
from torch import nn


class ProteinDataManager(DataManager):
    def __init__(self, train_dbids, dev_dbids, test_dbids, use_cuda, max_len):
        super(ProteinDataManager, self).__init__(train_dbids, dev_dbids, test_dbids, use_cuda)
        print('Initializing ProteinDataManager...')
        self.max_len = max_len
        self.vocab = vocab.Vocab()

        with open('/data/data_targetProteins/train_target_seq.pkl', 'rb') as infile:
            dbid_to_seqs = pickle.load(infile)
        with open('/data/data_targetProteins/test_target_seq.pkl', 'rb') as infile:
            dbid_to_seqs.update(pickle.load(infile))
        train_sents = [[self.vocab.numberize_sentence(seq) for seq in dbid_to_seqs[dbid]] for dbid in self.train_dbids]
        self.train_sents_tensor, self.train_sents_len, self.train_max_proteins = self.get_sents_tensor(train_sents)
        dev_sents = [[self.vocab.numberize_sentence(seq) for seq in dbid_to_seqs[dbid]] for dbid in self.dev_dbids]
        self.dev_sents_tensor, self.dev_sents_len, self.dev_max_proteins = self.get_sents_tensor(dev_sents)
        test_sents = [[self.vocab.numberize_sentence(seq) for seq in dbid_to_seqs[dbid]] for dbid in self.test_dbids]
        self.test_sents_tensor, self.test_sents_len, self.test_max_proteins = self.get_sents_tensor(test_sents)

        self.embed = None
        print('ProteinDataManager initialized.')

    def connect_to_submodule(self, submodule):
        nn.init.xavier_normal_(submodule.aa_embed.weight)
        self.embed = submodule.aa_embed

    def get_sents_tensor(self, sents_num):
        """ sents_num: list of lists of word ids """
        num_dbids = len(sents_num)
        max_num_proteins = max([len(protein_seqs) for protein_seqs in sents_num])
        r_tensor = torch.LongTensor(num_dbids * max_num_proteins, self.max_len)
        r_tensor.fill_(vocab.PAD_token)
        slen_tensor = torch.LongTensor(num_dbids * max_num_proteins, )
        if self.use_cuda:
            slen_tensor = slen_tensor.cuda()

        b = 0
        for dbid_seqs in sents_num:
            proteins_added = 0
            for protein_seq in dbid_seqs:
                slen = min(len(protein_seq), self.max_len)
                slen_tensor[b] = slen
                for w in range(slen):
                    wordid = protein_seq[w]
                    r_tensor[b, slen - 1 - w] = wordid  # fill in reverse
                b += 1
                proteins_added += 1
            while proteins_added < max_num_proteins:
                slen_tensor[b] = 1
                r_tensor[b, -1] = vocab.EOS_token
                b += 1
                proteins_added += 1
        return r_tensor, slen_tensor, max_num_proteins

    def sample_batch(self, dbids, dbid_to_idx, sents_tensor, sents_len, max_num_proteins):
        idx = []
        for dbid in dbids:
            idx += [dbid_to_idx[dbid] * max_num_proteins + i for i in range(max_num_proteins)]
        batch_sents_tensor = sents_tensor[idx]
        batch_sents_len = sents_len[idx]

        des_packed_tensor, des_idx_unsort = self.vocab.get_packedseq_from_sent_batch(
            seq_tensor=batch_sents_tensor,
            seq_lengths=batch_sents_len,
            embed=self.embed,
            use_cuda=self.use_cuda,
        )

        return (des_packed_tensor,
                des_idx_unsort,
                max_num_proteins,
                len(dbids),
                self.use_cuda)

    def sample_train_batch(self, dbids):
        return self.sample_batch(dbids, self.train_dbid_to_idx, self.train_sents_tensor, self.train_sents_len,
                                 self.train_max_proteins)

    def sample_dev_batch(self, dbids):
        return self.sample_batch(dbids, self.dev_dbid_to_idx, self.dev_sents_tensor, self.dev_sents_len,
                                 self.dev_max_proteins)

    def sample_test_batch(self, dbids):
        return self.sample_batch(dbids, self.test_dbid_to_idx, self.test_sents_tensor, self.test_sents_len,
                                 self.test_max_proteins)
