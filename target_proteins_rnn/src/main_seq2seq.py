import numpy as np
import torch
import pickle
import os
from train_seq2seq import trainIter
import hparams_seq2seq as hparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all

def main(train_seq, letter_to_idx):  
    model_trained, embeds_trained = trainIter(train_seq, letter_to_idx, hparams.N_EPOCHS, hparams.SAVE_PATH)
    return model_trained, embeds_trained


if __name__ == '__main__':
    # create output dir
#    output_dir = hparams.ROOT_PATH + 'output'
#    if not os.path.exists(hparams.ROOT_PATH + 'output'):
#        os.mkdir(output_dir)
        
    # load inputs
    with open(hparams.LETTER_TO_TEXT_FILE, 'rb') as f1:
        letter_to_idx = pickle.load(f1)
            
    with open(hparams.TRAIN_SEQ_FILE, 'rb') as f:
        train_seq = pickle.load(f) 
            
    # train
    model_trained, embeds_trained = main(train_seq, letter_to_idx)
              
    # save
    with open(hparams.SAVE_PATH + 'trained.pkl', 'wb') as f:
        pickle.dump(model_trained, f)
        pickle.dump(embeds_trained, f)
    
