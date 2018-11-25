import torch
import torch.nn as nn
import numpy as np
import src.dataman.vocab as vocab


class RNNEncoder(nn.Module):
    def __init__(self, config):
        super(RNNEncoder, self).__init__()
        self.config = config
        input_size = config.embedding_size
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layer,
            dropout=config.dp_ratio,
            bidirectional=config.bidirectional)

    def initHidden(self, batch_size, cuda):
        if self.config.bidirectional:
            state_shape = (2, batch_size, self.config.hidden_size)
        else:
            state_shape = (1, batch_size, self.config.hidden_size)
        if cuda:
            h0 = c0 = torch.cuda.FloatTensor(*state_shape).fill_(0)
        else:
            h0 = c0 = torch.zeros(*state_shape)
        return h0, c0

    def forward(self, inputs, hidden):
        outputs, (ht, ct) = self.rnn(inputs, hidden)
        return outputs


# class TextBaseModel(nn.Module):
#     def __init__(self, config, classifier):
#         super(TextBaseModel, self).__init__()
#         self.config = classifier.config
#         self.glove_embed = classifier.glove_embed
#         self.other_embed = classifier.other_embed
#         self.glove_embed.weight.requires_grad = False
#         self.other_embed.weight.requires_grad = False
#         self.encoder = classifier.encoder
#         self.base = classifier.base
#
#     def forward(
#         self,
#         des_embed,
#         des_unsort,
#         ind_embed,
#         ind_unsort,
#         act_embed,
#         act_unsort,
#         encoder_init_hidden,
#         batch_size
#     ):
#         des_rnn = self.encoder(
#                 inputs=des_embed,
#                 hidden=encoder_init_hidden,
#                 batch_size=batch_size
#             )
#         des_rnn = nn.utils.rnn.pad_packed_sequence(des_rnn, padding_value=-np.infty)[0]
#         des_rnn = des_rnn.index_select(1, des_unsort)
#         ind_rnn = self.encoder(
#             inputs=des_embed,
#             hidden=encoder_init_hidden,
#             batch_size=batch_size
#         )
#         ind_rnn = nn.utils.rnn.pad_packed_sequence(ind_rnn, padding_value=-np.infty)[0]
#         ind_rnn = ind_rnn.index_select(1, ind_unsort)
#         act_rnn = self.encoder(
#             inputs=act_embed,
#             hidden=encoder_init_hidden,
#             batch_size=batch_size
#         )
#         act_rnn = nn.utils.rnn.pad_packed_sequence(act_rnn, padding_value=-np.infty)[0]
#         act_rnn = act_rnn.index_select(1, act_unsort)
#
#         des_maxpool = torch.max(des_rnn, 0)[0]  # [batch_size, embed_size]
#         ind_maxpool = torch.max(ind_rnn, 0)[0]
#         act_maxpool = torch.max(act_rnn, 0)[0]
#
#         scores = self.base(torch.cat([
#             des_maxpool,
#             ind_maxpool,
#             act_maxpool # [batch_size, 3*embed_size]
#         ], 1))  # [batch_size, 3*last_hidden_size]
#
#         return scores
#
#
# class TextClassifier(nn.Module):
#     def __init__(self, config):
#         super(TextClassifier, self).__init__()
#         self.config = config
#         self.glove_embed = nn.Embedding(config.n_embed, config.glove_embedding_size, padding_idx=vocab.PAD_token)
#         self.other_embed = nn.Embedding(config.n_embed, config.other_embedding_size, padding_idx=vocab.PAD_token)
#         if self.config.fix_emb_glove:
#             self.glove_embed.weight.requires_grad = False
#         if self.config.fix_emb_other:
#             self.other_embed.weight.requires_grad = False
#
#         self.encoder = RNNEncoder(config)
#
#         self.dropout = nn.Dropout(p=config.dp_ratio)
#         self.relu = nn.ReLU()
#
#         hidden_size = config.hidden_size
#         if self.config.bidirectional:
#             hidden_size *= 2
#         hidden_size *= 3
#
#         mlp_layers = []
#         prev_hidden_size = hidden_size
#         for next_hidden_size in config.mlp_hidden_size_list:
#             mlp_layers.extend([
#                 nn.Linear(prev_hidden_size, next_hidden_size),
#                 self.relu,
#                 self.dropout,
#             ])
#             prev_hidden_size = next_hidden_size
#         self.base = nn.Sequential(*mlp_layers)
#         self.out = nn.Linear(prev_hidden_size, config.n_label)
#
#     def forward(
#         self,
#         des_embed,
#         des_unsort,
#         ind_embed,
#         ind_unsort,
#         act_embed,
#         act_unsort,
#         encoder_init_hidden,
#         batch_size
#     ):
#         des_rnn = self.encoder(
#                 inputs=des_embed,
#                 hidden=encoder_init_hidden,
#                 batch_size=batch_size
#             )
#         des_rnn = nn.utils.rnn.pad_packed_sequence(des_rnn, padding_value=-np.infty)[0]
#         des_rnn = des_rnn.index_select(1, des_unsort)
#         ind_rnn = self.encoder(
#             inputs=des_embed,
#             hidden=encoder_init_hidden,
#             batch_size=batch_size
#         )
#         ind_rnn = nn.utils.rnn.pad_packed_sequence(ind_rnn, padding_value=-np.infty)[0]
#         ind_rnn = ind_rnn.index_select(1, ind_unsort)
#         act_rnn = self.encoder(
#             inputs=act_embed,
#             hidden=encoder_init_hidden,
#             batch_size=batch_size
#         )
#         act_rnn = nn.utils.rnn.pad_packed_sequence(act_rnn, padding_value=-np.infty)[0]
#         act_rnn = act_rnn.index_select(1, act_unsort)
#
#         des_maxpool = torch.max(des_rnn, 0)[0]  # [batch_size, embed_size]
#         ind_maxpool = torch.max(ind_rnn, 0)[0]
#         act_maxpool = torch.max(act_rnn, 0)[0]
#
#         scores = self.base(torch.cat([
#             des_maxpool,
#             ind_maxpool,
#             act_maxpool  # [batch_size, 3*embed_size]
#         ], 1))  # [batch_size, 3*last_hidden_size]
#
#         logit_outputs = self.out(scores)  # [batch_size, n_label]
#         return logit_outputs


class TextEmbeddingModel(nn.Module):
    def __init__(self, config):
        super(TextEmbeddingModel, self).__init__()
        self.config = config
        self.glove_embed = nn.Embedding(config.n_embed, config.glove_embedding_size, padding_idx=vocab.PAD_token)
        self.other_embed = nn.Embedding(config.n_embed, config.other_embedding_size, padding_idx=vocab.PAD_token)
        if self.config.fix_emb_glove:
            self.glove_embed.weight.requires_grad = False
        if self.config.fix_emb_other:
            self.other_embed.weight.requires_grad = False

        self.encoder = RNNEncoder(config)

        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()

    def buh(self, lens):
        lens = lens - 1
        lens = lens.reshape(-1, 1, 1)
        lens = lens.repeat(1, 1, self.config.hidden_size)
        lens = lens.cuda()
        return lens

    def forward(
        self,
        des_embed,
        des_unsort,
        des_unsort_lens,
        ind_embed,
        ind_unsort,
        ind_unsort_lens,
        act_embed,
        act_unsort,
        act_unsort_lens,
        batch_size,
        cuda
    ):
        encoder_init_hidden = self.encoder.initHidden(batch_size, cuda)
        des_rnn = self.encoder(
                inputs=des_embed,
                hidden=encoder_init_hidden,
            )
        des_rnn, des_lens = nn.utils.rnn.pad_packed_sequence(des_rnn, padding_value=-np.infty)
        # des_index = self.buh(des_lens)
        # des_maxpool = torch.gather(des_rnn.transpose(0, 1), 1, des_index).squeeze(0)
        # des_maxpool = des_maxpool.index_select(1, des_unsort)
        des_rnn = des_rnn.index_select(1, des_unsort)

        ind_rnn = self.encoder(
            inputs=ind_embed,
            hidden=encoder_init_hidden,
        )
        ind_rnn, ind_lens = nn.utils.rnn.pad_packed_sequence(ind_rnn, padding_value=-np.infty)
        # ind_index = self.buh(ind_lens)
        # ind_maxpool = torch.gather(ind_rnn.transpose(0, 1), 1, ind_index).squeeze(0)
        # ind_maxpool = ind_maxpool.index_select(1, ind_unsort)
        ind_rnn = ind_rnn.index_select(1, ind_unsort)

        act_rnn = self.encoder(
            inputs=act_embed,
            hidden=encoder_init_hidden,
        )
        act_rnn, act_lens = nn.utils.rnn.pad_packed_sequence(act_rnn, padding_value=-np.infty)
        # act_index = self.buh(act_lens)
        # act_maxpool = torch.gather(act_rnn.transpose(0, 1), 1, act_index).squeeze(1)
        # act_maxpool = act_maxpool.index_select(1, act_unsort)
        act_rnn = act_rnn.index_select(1, act_unsort)

        des_maxpool = torch.max(des_rnn, 0)[0]  # [batch_size, embed_size]
        ind_maxpool = torch.max(ind_rnn, 0)[0]
        act_maxpool = torch.max(act_rnn, 0)[0]
        # des_maxpool = torch.gather(des_rnn, 0,
        #                            (des_unsort_lens - 1).view(1, -1)
        #                            .unsqueeze(2).repeat(1, 1, self.config.hidden_size)).squeeze(0)
        # ind_maxpool = torch.gather(ind_rnn, 0,
        #                            (ind_unsort_lens - 1).view(1, -1)
        #                            .unsqueeze(2).repeat(1, 1, self.config.hidden_size)).squeeze(0)
        # act_maxpool = torch.gather(act_rnn, 0,
        #                            (act_unsort_lens - 1).view(1, -1)
        #                            .unsqueeze(2).repeat(1, 1, self.config.hidden_size)).squeeze(0)

        scores = torch.cat([des_maxpool, ind_maxpool, act_maxpool], dim=1)

        # scores = self.base(torch.cat([
        #     des_maxpool,
        #     ind_maxpool,
        #     act_maxpool  # [batch_size, 3*embed_size]
        # ], 1))  # [batch_size, 3*last_hidden_size]

        return scores
