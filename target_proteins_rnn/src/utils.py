import numpy as np
import torch
import matplotlib.pyplot as plt
import hparams_seq2seq as hparams_seq2seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
torch.cuda.manual_seed_all


def seq2letter(sequence):
    seq_letters = []
    for seq in sequence:
        seq_letters.append(seq)
    return seq_letters

def prepare_seq(seq, letter_to_idx):
    indices = []
    for w in seq:
        if w not in letter_to_idx: # if not found, map to unknown token
            indices.append(hparams_seq2seq.UNK_token)          
        else:
            indices.append(letter_to_idx[w])
    indices.append(hparams_seq2seq.EOS_token)
    return torch.tensor(indices, dtype=torch.long, device=device)


def prepare_label(label, vocab):
    return torch.tensor([vocab[label]], dtype=torch.long, device=device)


def go2words(go):
    words = go.split(' ')
    for idx, word in enumerate(words):
        word = word.replace("[", "")
        word = word.replace("]", "")
        word = word.replace(",","")
        words[idx] = word
    return words


def aggregateSeq(all_embeds, aggregate='MEAN', keep_dim=False):
    if aggregate == 'MAX':
        maxes, idxs = torch.max(all_embeds, 0, keepdim=keep_dim)
        return maxes
    elif aggregate == 'SUM':  
        return torch.sum(all_embeds, 0, keepdim=keep_dim)
    else:
        return torch.mean(all_embeds, 0, keepdim=keep_dim)


def plotTrainValLoss(train_loss, val_loss, save_path):
    plt.figure()
    x = np.arange(len(train_loss))
    plt.plot(x, train_loss)
    plt.plot(x, val_loss)
    plt.legend(['train loss', 'val loss'], loc = 'upper right')
    plt.savefig(save_path + 'train_losses.png')
#    plt.show()
                




    
