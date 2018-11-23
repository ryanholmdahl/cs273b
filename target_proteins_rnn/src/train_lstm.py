import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
from utils import go2words, prepare_label, prepare_seq, aggregateSeq, plotTrainValLoss
import hparams_lstm as hparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed_all
np.random.seed(1)


def train(train_data, train_label, lstm_model, lstm_optimizer, criterion):
    lstm_model.hidden = lstm_model.init_hidden()
    lstm_model.zero_grad()
            
    lstm_out, last_hidden = lstm_model(train_data)
    loss = criterion(lstm_out, train_label)    
    
    loss.backward()
    
    lstm_optimizer.step()
    
    return loss


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
#        print("lstm_out:{}".format(lstm_out))
#        print("val_label:{}".format(val_label))
        loss = criterion(lstm_out, val_label)
#        print(loss)
        acc = compute_acc(val_label, pred_label)
    return loss, acc


def trainIter(lstm_model, train_goterms, gowords_vocab, goterms_vocab, n_epochs, save_path):
    criterion = nn.NLLLoss()
    lstm_optimizer = optim.SGD(lstm_model.parameters(), lr = hparams.LEARNING_RATE, momentum = 0.9)
    # prepare inputs
    train_gowords = []
    for go in train_goterms:
        train_gowords.append(go2words(go))
    
    # split train & validation sets
    indices = np.random.permutation(len(train_gowords))  
    bound = int(round(len(train_gowords)*0.8))
    train_idx = indices[:bound]
    val_idx = indices[bound:]
    train_data = [train_gowords[i] for i in train_idx]
    train_label = [train_goterms[i] for i in train_idx]
    val_data = [train_gowords[i] for i in val_idx]
    val_label = [train_goterms[i] for i in val_idx]
    
    best_val_acc = 0.0
    plot_losses_train = []
    plot_losses_val = []
    for epoch in range(hparams.N_EPOCHS):
        print("Current epoch:{}".format(epoch))
        # train
        train_loss_total = 0.0
        for idx, goword in enumerate(train_data):
            # prepare data
            goterm = train_label[idx]
            input_tensor = prepare_seq(goword, gowords_vocab)
            label_tensor = prepare_label(goterm, goterms_vocab)
            
            loss = train(input_tensor, label_tensor, lstm_model, lstm_optimizer, criterion)
#            print("Current loss:{}".format(loss))
            train_loss_total += loss
        avg_train_loss = train_loss_total / len(train_data)
        plot_losses_train.append(avg_train_loss)
        
        # evaluate
        val_loss_total = 0.0
        val_acc_total = 0.0
        for idx, goword in enumerate(val_data):
            goterm = val_label[idx]
            input_tensor = prepare_seq(goword, gowords_vocab)
            label_tensor = prepare_label(goterm, goterms_vocab)
            loss, acc = evaluate(input_tensor, label_tensor, lstm_model, criterion)
            val_loss_total += loss
            val_acc_total += acc
        val_loss = val_loss_total / len(val_data)
        val_acc = val_acc_total / len(val_data)
        plot_losses_val.append(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.system('rm ' + save_path + 'bestModel_valAcc*.model')
            print('New Best Val Acc!')
            torch.save(lstm_model.state_dict(), save_path + 'bestModel_valAcc' + str(int(best_val_acc*10000)) + '.model')
                
              
#    plotTrainValLoss(plot_losses_train, plot_losses_val, save_path)
    
    return lstm_model, best_val_acc
        
        

def get_trained_embeds(goterms_dict, gowords_to_idx, lstm_model):
    
    embeds_trained = []
    with torch.no_grad():
        for idx_drug in range(len(goterms_dict)):
            drug_goterms = goterms_dict[idx_drug]
    
            if len(drug_goterms) == 0:
                aggreg_embeds = torch.zeros(hparams.HIDDEN_SIZE, device=device)
            else:
                lstm_model.hidden = lstm_model.init_hidden()
                all_embeds = torch.zeros(0, 0, device=device)
                for idx_targ in range(len(drug_goterms)):
                    curr_go = drug_goterms[idx_targ]
                    goterms = ""
                    for i in range(len(curr_go)):
                        goterms = goterms + curr_go[i]
                    input_gowords = go2words(goterms)
                    input_tensor = prepare_seq(input_gowords, gowords_to_idx)
                    output, last_hidden = lstm_model(input_tensor)
                    last_hidden = last_hidden[0][0].view(1,-1)
#                    last_hidden = lstm_model.hidden[-1][0][0].view(1,-1)
                    all_embeds = torch.cat((all_embeds, last_hidden), dim = 0)
#                aggreg_embeds = aggregateSeq(all_embeds, 'MEAN', keep_dim=False)
                aggreg_embeds = lstm_model.aggregate(all_embeds, 'MEAN', keep_dim=False)
            embeds_trained.append(aggreg_embeds)
            
        embeds_trained = torch.stack(embeds_trained)
            
    return embeds_trained            
            
            
            
            
            
            
            
            
            
            
            
            
            
