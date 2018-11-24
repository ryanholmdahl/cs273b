import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from model_lstm import gotermsClassifier
import hparams_lstm as hparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed_all
np.random.seed(1)

def compute_acc(y_true, y_pred):
    assert len(y_true)==len(y_pred)
    correct = 0
    for i in range(len(y_true)):
        if y_true[i]==y_pred[i]:
            correct += 1.0
    return correct/len(y_true)
    

def evaluate(val_data, val_label, lstm_model, criterion):
    with torch.no_grad():
        lstm_model.hidden = lstm_model.init_hidden()
        lstm_out, last_hidden = lstm_model(val_data)
        pred_label = lstm_out.data.max(1)[1]
        loss = criterion(lstm_out, val_label)
        print(loss)
        acc = compute_acc(val_label, pred_label)
    return loss, acc


def trainIter(train_goterms, gowords_vocab, goterms_vocab, labels, n_epochs, save_path):
    criterion = nn.NLLLoss()
    config = {'embedding_size': hparams.EMBEDDING_DIM,
                    'hidden_size': hparams.HIDDEN_SIZE,
                    'vocab_size': len(gowords_vocab),
                    'label_size': len(goterms_vocab),
                    'lr': hparams.LEARNING_RATE
                    }
                     
    lstm_model = gotermsClassifier(config, criterion)
    optimizer = optim.SGD(lstm_model.parameters(), lr = hparams.LEARNING_RATE, momentum = 0.9)

    for epoch in range(hparams.N_EPOCHS):  
        lstm_model.train()
        print("Current epoch:{}".format(epoch))
        # train
        all_embeds = []
        train_losses = 0.0
        for dbid, drug_goterms in train_goterms.items():
            curr_labels = labels[dbid]
            if len(drug_goterms) == 0 or (len(drug_goterms[0]) == 0):
                curr_embeds = torch.zeros(hparams.HIDDEN_SIZE, device=device)
            else:
                curr_embeds, loss = lstm_model(drug_goterms, gowords_vocab, curr_labels)              
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses += loss
                
            all_embeds.append(curr_embeds)
     
        avg_train_loss = train_losses / len(train_goterms)
        all_embeds = torch.stack(all_embeds)

#        print("Training loss in current epoch:{}".format(avg_train_loss))

    return lstm_model, all_embeds
            
            
            
            
            
            
            
            
            
            
            
            
            
