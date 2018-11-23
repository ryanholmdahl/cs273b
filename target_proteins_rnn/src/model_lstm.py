import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class lstmClassifier(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super(lstmClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim, device=device),
                torch.zeros(1, 1, self.hidden_dim, device=device))
         
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y, dim=1)
        last_hidden = self.hidden[-1]
        return log_probs, last_hidden
    
    def aggregate(self, last_hiddens, aggregate='MEAN', keep_dim=False):
        if aggregate == 'MAX':
            maxes, idxs = torch.max(last_hiddens, 0, keepdim=keep_dim)
            return maxes
        elif aggregate == 'SUM':  
            return torch.sum(last_hiddens, 0, keepdim=keep_dim)
        else:
            return torch.mean(last_hiddens, 0, keepdim=keep_dim)
    
    
    
    
    
    
    
    
    
    