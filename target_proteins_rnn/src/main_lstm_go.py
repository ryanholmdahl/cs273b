import numpy as np
import torch
import pickle
import itertools as itools
import os
from model_lstm import lstmClassifier
from train_lstm import get_trained_embeds, trainIter
import hparams_lstm as hparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all


def main(train_goterms_1d, train_goterms, gowords_vocab, goterms_vocab):
    
    lstm_model = lstmClassifier(hparams.EMBEDDING_DIM, hparams.HIDDEN_SIZE, 
                                len(gowords_vocab), len(goterms_vocab))
            
    lstm_trained, avg_val_acc = trainIter(lstm_model, train_goterms_1d, gowords_vocab, goterms_vocab, 
                                          hparams.N_EPOCHS, hparams.SAVE_PATH_GO)
    embeds_trained = get_trained_embeds(train_goterms, gowords_vocab, lstm_trained)
    return lstm_trained, embeds_trained, avg_val_acc


if __name__ == '__main__':
    # create output dir
    output_dir = hparams.ROOT_PATH + 'output'
    if not os.path.exists(hparams.ROOT_PATH + 'output'):
        os.makedirs(output_dir)
        
    # load inputs
    with open(hparams.GOWORDS_VOCAB_FILE, 'rb') as f1:
        gowords_vocab = pickle.load(f1)
            
    with open(hparams.GOTERMS_VOCAB_FILE, 'rb') as f:
        goterms_vocab = pickle.load(f) 
        
    with open(hparams.TRAIN_GOTERMS_FILE, 'rb') as f:
        train_goterms = pickle.load(f)
        
    train_goterms_1d = list(itools.chain.from_iterable(itools.chain.from_iterable(train_goterms)))
    
    # train
    lstm_trained, embeds_trained, avg_val_acc = main(train_goterms_1d, train_goterms, gowords_vocab, goterms_vocab)
    
    # save
    with open(hparams.SAVE_PATH_GO + 'trained.pkl', 'wb') as f:
        pickle.dump(lstm_trained, f)
        pickle.dump(embeds_trained, f)
        pickle.dump(avg_val_acc, f)
    
    

