import torch
import torch.nn as nn
#import torch.autograd as autograd
import torch.nn.functional as F
import hparams_lstm as hparams
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
    
    
class gotermsClassifier(nn.Module):
    
    def __init__(self, config, criterion):
        super(gotermsClassifier, self).__init__()
        self.hidden_size = config['hidden_size']
        self.embedding_size = config['embedding_size']
        self.vocab_size = config['vocab_size']
        self.label_size = config['label_size']
        self.criterion = criterion
        self.model = lstmClassifier(self.embedding_size, self.hidden_size, 
                                    self.vocab_size, self.label_size).to(device)
        
    def forward(self, drug_goterms, vocab, labels): # drug_goterms is a list of the GO terms for a particular drug
        loss = 0.0
        ct = 0.0
        all_embeds = torch.zeros(0, 0, device=device)
        for idx_targ in range(len(drug_goterms)):
            targ_input = drug_goterms[idx_targ]
            targ_label = labels[idx_targ]
            if isinstance(targ_label,int):
                input_tensor = self.prepare_seq(self.go2words(targ_input), vocab)
                label_tensor = torch.tensor([targ_label], dtype=torch.long, device=device)
                
                self.model.hidden = self.model.init_hidden()
                
                output, last_hidden = self.model(input_tensor)
                loss += self.criterion(output, label_tensor) 
                
                last_hidden = last_hidden[0][0].view(1,-1)
                all_embeds = torch.cat((all_embeds, last_hidden), dim=0)
                ct += 1
            else:
                for i in range(len(targ_input)):
                    input_gowords = self.go2words(targ_input[i])
                    input_tensor = self.prepare_seq(input_gowords, vocab)
                    label_tensor = torch.tensor([targ_label[i]], dtype=torch.long, device=device)
                    
                    self.model.hidden = self.model.init_hidden()
            
                    output, last_hidden = self.model(input_tensor)
                    loss += self.criterion(output, label_tensor) 
                                            
                    last_hidden = last_hidden[0][0].view(1,-1)
                    all_embeds = torch.cat((all_embeds, last_hidden), dim = 0)
                    ct += 1
                        
        embeds = self.aggregate(all_embeds, 'MEAN', keep_dim=False)
        return embeds, loss/ct
        
    
    def aggregate(self, last_hiddens, aggregate='MEAN', keep_dim=False):
        if aggregate == 'MAX':
            maxes, idxs = torch.max(last_hiddens, 0, keepdim=keep_dim)
            return maxes
        elif aggregate == 'SUM':  
            return torch.sum(last_hiddens, 0, keepdim=keep_dim)
        else:
            return torch.mean(last_hiddens, 0, keepdim=keep_dim)

       
    def prepare_seq(self, seq, letter_to_idx):
        indices = []
        for w in seq:
            if w not in letter_to_idx: # if not found, map to unknown token
                indices.append(hparams.UNK_token)          
            else:
                indices.append(letter_to_idx[w])
        indices.append(hparams.EOS_token)
        return torch.tensor(indices, dtype=torch.long, device=device)


    def prepare_label(self, label, vocab):
        return torch.tensor([vocab[label]], dtype=torch.long, device=device)
    
    
    def go2words(self, go):
        words = go.split(' ')
        for idx, word in enumerate(words):
            word = word.replace("[", "")
            word = word.replace("]", "")
            word = word.replace(",","")
            words[idx] = word
        return words
    
