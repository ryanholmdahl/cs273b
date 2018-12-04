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


class ProteinEmbeddingModel(nn.Module):
    def __init__(self, config):
        super(ProteinEmbeddingModel, self).__init__()
        self.config = config
        self.aa_embed = nn.Embedding(config.n_embed, config.embedding_size, padding_idx=vocab.PAD_token)
        self.encoder = RNNEncoder(config)
        self.file_name = 'protein.pt'

    def forward(
        self,
        protein_embed,  # [batch_size * proteins_per, max_protein_len]
        protein_unsort,
        proteins_per,
        batch_size,
        cuda
    ):
        encoder_init_hidden = self.encoder.initHidden(batch_size * proteins_per, cuda)
        protein_rnn = self.encoder(
                inputs=protein_embed,
                hidden=encoder_init_hidden,
            )
        protein_rnn, _ = nn.utils.rnn.pad_packed_sequence(protein_rnn, padding_value=-np.infty)
        protein_rnn = protein_rnn.index_select(1, protein_unsort)  # [batch_size * proteins_per, max_protein_len]
        protein_rnn = protein_rnn.reshape(self.config.max_len, batch_size, proteins_per, -1)
        protein_maxpool = torch.max(protein_rnn, 0)[0]
        protein_maxpool = torch.max(protein_maxpool, dim=1)[0]  # [batch_size, embed_size]

        return protein_maxpool
