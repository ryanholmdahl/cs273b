import numpy as np
import torch
import pickle
import os
from train_lstm import trainIter
import hparams_lstm as hparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all


def main(train_goterms, gowords_vocab, goterms_vocab, labels):
    
    lstm_trained, embeds = trainIter(train_goterms, gowords_vocab, goterms_vocab, labels, hparams.N_EPOCHS, hparams.SAVE_PATH_GO)            

    return lstm_trained, embeds


if __name__ == '__main__':
    # create output dir
#    output_dir = hparams.ROOT_PATH + 'output'
#    if not os.path.exists(hparams.ROOT_PATH + 'output'):
#        os.mkdir(hparams.ROOT_PATH + 'output')
        
    # load inputs
    with open(hparams.GOWORDS_VOCAB_FILE, 'rb') as f1:
        gowords_vocab = pickle.load(f1)
            
    with open(hparams.GOTERMS_VOCAB_FILE, 'rb') as f:
        goterms_vocab = pickle.load(f) 
        
    with open(hparams.TRAIN_GOTERMS_FILE, 'rb') as f:
        train_goterms = pickle.load(f)
        
    with open(hparams.GOTERMS_LABEL_FILE, 'rb') as f:
        labels = pickle.load(f)
            
    # train
    lstm_trained, embeds_trained = main(train_goterms, gowords_vocab, goterms_vocab, labels)
    
    # save
    with open(hparams.SAVE_PATH_GO + 'trained.pkl', 'wb') as f:
        pickle.dump(lstm_trained, f)
        pickle.dump(embeds_trained, f)
f    
    

