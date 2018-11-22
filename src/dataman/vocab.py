# python 3
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
import numpy as np
from nltk.tokenize import word_tokenize
import re
import io
import array
import logging
import pickle
from gensim.models import KeyedVectors
from tqdm import tqdm

logger = logging.getLogger(__name__)

PAD_token = 0
EOS_token = 1
UNK_token = 2

splitstring = '-|/|_|\+|@|\.|,'

class Vocab:
    def __init__(self, logdir=None):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: '<pad>', 1: "EOS", 2: "UNK"}
        self.n_words = 3
        self.glove_tensors = None
        self.glove_itos = None
        self.glove_stoi = None
        self.glove_dim = None
        self.medw2v_model = None
        self.medw2v_dim = None
        if logdir and os.path.isdir(logdir):
            with open(os.path.join(logdir, 'text_idx2word.pkl'), 'rb') as fd:
                self.index2word = pickle.load(fd)
            with open(os.path.join(logdir, 'text_word2cnt.pkl'), 'rb') as fd:
                self.word2count = pickle.load(fd)
            with open(os.path.join(logdir, 'text_word2idx.pkl'), 'rb') as fd:
                self.word2index = pickle.load(fd)
            self.n_words = len(self.index2word)

    def addSentence(self, sentence):
        if sentence is None:
            sentence = ""  # LSTM requires seq_len > 0
        sentence = sentence.strip().lower()
        words = word_tokenize(sentence)
        tokens = [x for longword in words if len(longword)>1 for x in re.split(splitstring, longword)]
        for word in tokens:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def numberize_sentence(self, sentence):
        if sentence is None:
            return [EOS_token]
        sentence = sentence.strip().lower()
        words = word_tokenize(sentence)
        tokens = [x for longword in words for x in re.split(splitstring, longword)]
        numberized = [
            self.word2index[word]
            if word in self.word2index else UNK_token
            for word in tokens
        ]
        numberized.append(EOS_token)
        return numberized

    def get_packedseq_from_sent_batch(
        self,
        seq_tensor,
        seq_lengths,
        embed1,
        embed2,
        use_cuda,
    ):
        if use_cuda:
            seq_lengths = seq_lengths.cuda()
            seq_tensor = seq_tensor.cuda()

        # sort by length
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        seq_tensor = seq_tensor.transpose(0, 1)  # [seq_len, batch_size]
        idx_unsort = Variable(torch.from_numpy(
            np.argsort(perm_idx.cpu().numpy())
        ))
        if use_cuda:
            idx_unsort = idx_unsort.cuda()

        # embed
        seq_tensor = torch.cat([embed1(seq_tensor),embed2(seq_tensor)],-1)

        # if torch.min(seq_lengths).item() <= 0:
        #     print(seq_lengths,seq_tensor)

        # pack
        seq_pack_tensor = nn.utils.rnn.pack_padded_sequence(
            seq_tensor.cuda() if use_cuda else seq_tensor,
            seq_lengths.cpu().numpy(),
        )

        return seq_pack_tensor, idx_unsort

    def load_glove(self, path, d_embed, size='6B'): # size can be '6B' or '840B'
        name = 'glove.' + size + '.' + str(d_embed) + 'd.txt'
        name_pt = name + '.pt'
        path_pt = os.path.join(path, name_pt)

        # load Glove
        if os.path.isfile(path_pt):  # Load .pt file if there is any cached
            print('Loading vectors from {}'.format(path_pt))
            itos, stoi, vectors, dim = torch.load(path_pt)
        else:  # Read from Glove .txt file
            path = os.path.join(path, name)
            if not os.path.isfile(path):
                raise RuntimeError('No files found at {}'.format(path))
            try:
                with io.open(path, encoding="utf8") as f:
                    lines = [line for line in f]
            except:
                raise RuntimeError('Could not read {} as format UTF8'.format(path))
            print("Loading vectors from {}".format(path))

            itos, vectors, dim = [], array.array(str('d')), None

            for line in tqdm(lines, total=len(lines)):
                entries = line.rstrip().split(" ")
                word, entries = entries[0], entries[1:]
                if dim is None and len(entries) > 1:
                    dim = len(entries)
                elif len(entries) == 1:
                    logger.warning("Skipping token {} with 1-dimensional "
                                   "vector {}; likely a header".format(word, entries))
                    continue
                elif dim != len(entries):
                    raise RuntimeError(
                        "Vector for token {} has {} dimensions, but previously "
                        "read vectors have {} dimensions. All vectors must have "
                        "the same number of dimensions.".format(word, len(entries), dim))
                vectors.extend(float(x) for x in entries)
                itos.append(word)
            stoi = {word: i for i, word in enumerate(itos)}
            vectors = torch.Tensor(vectors).view(-1, dim)
            print(vectors.shape)
            print('Saving vectors to {}'.format(path_pt))
            torch.save((itos, stoi, vectors, dim), path_pt)
        self.glove_tensors = vectors
        self.glove_itos = itos
        self.glove_stoi = stoi
        self.glove_dim = dim

    def add_glove_to_vocab(self, path, d_embed):
        if self.glove_itos is None:
            self.load_glove(path, d_embed)
        # Add it into vocab
        print('Adding GloVe words into vocab')
        for i, word in enumerate(self.glove_itos):
            self.addWord(word)

    def get_glove_embed_vectors(self, unk_init=torch.Tensor.zero_):
        # Look up vectors for words in vocab in the pretrained vectors
        vocab_tensors = torch.Tensor(self.n_words, self.glove_dim).zero_()
        # print(vocab.index2word)
        missing_cnt = 0
        for i, token in self.index2word.items():
            if i < 3:  # Skip the first 3 words PAD EOS UNK
                continue
            if token in self.glove_stoi:
                vocab_tensors[i][:] = self.glove_tensors[self.glove_stoi[token]]
            else:
                missing_cnt += 1
                vocab_tensors[i][:] = unk_init(torch.Tensor(1, self.glove_dim))
        #print('GloVe Miss', missing_cnt, 'words')
        return vocab_tensors.cpu().numpy()

    def load_medw2v(self, path):
        name = 'wikipedia-pubmed-and-PMC-w2v.bin'
        path = os.path.join(path, name)
        print('Loading W2V from {}'.format(path))
        self.medw2v_model = KeyedVectors.load_word2vec_format(path, binary=True)
        self.medw2v_dim = self.medw2v_model['a'].shape[0]

    def add_medw2v_to_vocab(self, path):
        if self.medw2v_model is None:
            self.load_medw2v(path)
        # Add it into vocab
        print('Adding PubMed words into vocab')
        for word in self.medw2v_model.index2entity:
            self.addWord(word)

    def get_medw2v_embed_vectors(self, unk_init=torch.Tensor.zero_):
        # Look up vectors for words in vocab in the pretrained vectors
        vocab_tensors = torch.Tensor(self.n_words, self.medw2v_dim).zero_()
        # print(vocab.index2word)
        missing_cnt = 0
        for i, token in self.index2word.items():
            if i < 3:  # Skip the first 3 words PAD EOS UNK
                continue
            try:
                vocab_tensors[i][:] = torch.Tensor(self.medw2v_model[token]).view(-1, self.medw2v_dim)
            except KeyError:
                missing_cnt += 1
                vocab_tensors[i][:] = unk_init(torch.Tensor(1, self.glove_dim))
        #print('PubMed-w2v Miss', missing_cnt, 'words')
        return vocab_tensors.cpu().numpy()