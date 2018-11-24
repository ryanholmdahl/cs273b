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


def main(train_target_ids, target_letter_vocab, target_vocab, labels):
    
    lstm_trained, embeds = trainIter(train_target_ids, target_letter_vocab, target_vocab, labels, hparams.N_EPOCHS, hparams.SAVE_PATH_GO)            
    
    return lstm_trained, embeds


if __name__ == '__main__':
    # create output dir
#    output_dir = hparams.ROOT_PATH + 'output'
#    if not os.path.exists(hparams.ROOT_PATH + 'output'):
#        os.mkdir(output_dir)
        
    # load inputs
    with open(hparams.TRAIN_TARGET_FILE, 'rb') as f:
        train_target_ids = pickle.load(f)
    
    with open(hparams.TARGET_LABEL_FILE, 'rb') as f:
        labels = pickle.load(f)
        
    with open(hparams.TARGET_LETTER_VOCAB_FILE, 'rb') as f:
        target_letter_vocab = pickle.load(f)
            
    with open(hparams.TARGET_VOCAB_FILE, 'rb') as f:
        target_vocab = pickle.load(f)
    
    # train
    model_trained, embeds_trained = main(train_target_ids, target_letter_vocab, target_vocab, labels)
    
    # save
    with open(hparams.SAVE_PATH_TARGET + 'trained.pkl', 'wb') as f:
        pickle.dump(model_trained, f)
        pickle.dump(embeds_trained, f)
    
    

