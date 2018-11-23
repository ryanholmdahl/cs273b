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


def main(targets_1d, target_ids, target_vocab, label_to_idx):
    
    lstm_model = lstmClassifier(hparams.EMBEDDING_DIM, hparams.HIDDEN_SIZE, 
                                len(target_vocab), len(label_to_idx)).to(device)
            
    lstm_trained, avg_val_acc = trainIter(lstm_model, targets_1d, target_vocab, label_to_idx, 
                                          hparams.N_EPOCHS, hparams.SAVE_PATH_TARGET)
    embeds_trained = get_trained_embeds(target_ids, target_vocab, lstm_trained)
    return lstm_trained, embeds_trained, avg_val_acc


if __name__ == '__main__':
    # create output dir
    output_dir = hparams.ROOT_PATH + 'output'
    if not os.path.exists(hparams.ROOT_PATH + 'output'):
        os.makedirs(output_dir)
        
    # load inputs
    with open(hparams.TARGET_VOCAB_FILE, 'rb') as f1:
        target_vocab = pickle.load(f1)
            
    with open(hparams.LABEL2IDX_FILE, 'rb') as f:
        label_to_idx = pickle.load(f) 
        
    with open(hparams.TRAIN_TARGET_FILE, 'rb') as f:
        train_target_ids = pickle.load(f)
        
    train_target_ids_1d = list(itools.chain.from_iterable(train_target_ids))
    
    # train
    lstm_trained, embeds_trained, avg_val_acc = main(train_target_ids_1d, train_target_ids,
                                                     target_vocab, label_to_idx)
    
    # save
    with open(hparams.SAVE_PATH_TARGET + 'trained.pkl', 'wb') as f:
        pickle.dump(lstm_trained, f)
        pickle.dump(embeds_trained, f)
        pickle.dump(avg_val_acc, f)
    
    

