import torch
import torch.nn as nn
import torch.nn.functional as F
import hparams_seq2seq as hparams
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed_all

class EncoderRNN(nn.Module):
    ''' For one target protein sequence
    '''
    def __init__(self, vocab_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        self.gru = nn.GRU(hidden_size, hidden_size)
        
        self.hidden = self.init_hidden()
                
    def init_hidden(self, cuda=False):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    def forward(self, sequence):
#        embeds = self.word_embeddings(sequence)
#        lstm_out, self.hidden = self.lstm(embeds.view(len(sequence), 1, -1), self.hidden)
        
        embedded = self.word_embeddings(sequence).view(1,1,-1)
        output = embedded
        output, self.hidden = self.gru(output, self.hidden)
        last_hidden = self.hidden[-1]
        return output, last_hidden
                    
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.word_embeddings = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.hidden = self.init_hidden()
        
    def init_hidden(self, cuda=False):
        return torch.zeros(1, 1, self.hidden_size, device=device)   
    
    def forward(self, sequence):
        output = self.word_embeddings(sequence).view(1,1,-1)
        output = F.relu(output)
        output, self.hidden = self.gru(output,self.hidden)
        output = self.softmax(self.out(output[0]))
        return output
    
    
class seqAutoencoder(nn.Module):
    def __init__(self, config, criterion):
        super(seqAutoencoder, self).__init__()
        self.vocab_size = config['vocab_size']
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_size']
        self.max_length = config['max_length']
        self.teacher_enforcing = config['teacher_enforcing']
        self.encoder = EncoderRNN(self.vocab_size, self.hidden_size).to(device)
        self.decoder = DecoderRNN(self.hidden_size, self.output_size).to(device)
        self.criterion = criterion
        
    def forward(self, drug_seq, letter_to_idx):
        loss = 0.0
        all_embeds = torch.zeros(0, 0, device=device)
        ct = 0
        for idx_targ in range(len(drug_seq)):
            curr_seq = drug_seq[idx_targ]
            input_letters = self.seq2letter(curr_seq)
            input_tensor = self.prepare_seq(input_letters, letter_to_idx)
            target_tensor = input_tensor # in our case, we will just use the input as the targets
            
            self.encoder.hidden = self.encoder.init_hidden()
            self.decoder.hidden = self.decoder.init_hidden()
            
            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)
            
            encoder_outputs = torch.zeros(self.max_length, self.hidden_size, device=device)
            
            # encoder
            for ei in range(input_length):
                encoder_output, last_hidden = self.encoder(torch.tensor([input_tensor[ei]], device=device))
                encoder_outputs[ei] = encoder_output[0,0]
            all_embeds = torch.cat((all_embeds, last_hidden), dim = 0)
            
            # decoder
            decoder_input = torch.tensor([[hparams.SOS_token]], device=device)
            self.decoder.hidden = self.encoder.hidden               
            if self.teacher_enforcing: # with teacher forcing
                for di in range(target_length):
                    decoder_output = self.decoder(decoder_input)
                    loss += self.criterion(decoder_output, torch.tensor([target_tensor[di]], device=device))
                    ct += 1
                    decoder_input = target_tensor[di] # teacher forcing, feed target as the next input
            else: # without teacher forcing   
                for di in range(target_length):
                    decoder_output = self.decoder(decoder_input)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()
                    
                    loss += self.criterion(decoder_output, torch.tensor([target_tensor[di]], device=device))
                    ct += 1
                    if decoder_input.item() == hparams.EOS_token:
                        break
                        
        # aggregate embeddings of all sequences of one drug into one embedding
        embeds = self.aggregate(all_embeds, 'MEAN', keep_dim=False)
        return embeds, loss / ct
        
              
    def aggregate(self, last_hiddens, aggregate='MEAN', keep_dim=False):
        if aggregate == 'MAX':
            maxes, idxs = torch.max(last_hiddens, 0, keepdim=keep_dim)
            return maxes
        elif aggregate == 'SUM':  
            return torch.sum(last_hiddens, 0, keepdim=keep_dim)
        else:
            return torch.mean(last_hiddens, 0, keepdim=keep_dim)
        
        
    def seq2letter(self, sequence):
        seq_letters = []
        for seq in sequence:
            seq_letters.append(seq)
        return seq_letters
    
    
    def prepare_seq(self, seq, letter_to_idx):
        indices = []
        for w in seq:
            if w not in letter_to_idx: # if not found, map to unknown token
                indices.append(hparams.UNK_token)          
            else:
                indices.append(letter_to_idx[w])
        indices.append(hparams.EOS_token)
        return torch.tensor(indices, dtype=torch.long, device=device)
        