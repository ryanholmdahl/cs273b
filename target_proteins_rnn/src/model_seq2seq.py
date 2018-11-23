import torch
import torch.nn as nn
import torch.nn.functional as F
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
        return output
    
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