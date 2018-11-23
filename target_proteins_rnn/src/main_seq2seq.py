import numpy as np
import torch
import pickle
import itertools as itools
import os
from model_seq2seq import EncoderRNN, DecoderRNN
from train_seq2seq import get_trained_embeds, trainIters
import hparams_seq2seq as hparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all

def main(train_seq_1d, train_seq, letter_to_idx, hidden_size):
    encoder = EncoderRNN(len(letter_to_idx), hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, len(letter_to_idx)).to(device)
    
    encoder_trained, decoder_trained, eval_dist = trainIters(train_seq_1d, letter_to_idx, 
                                                                    encoder, decoder, hparams.N_EPOCHS)
    seq_embed_trained = get_trained_embeds(train_seq, letter_to_idx, encoder_trained)
    
    return encoder_trained, decoder_trained, seq_embed_trained, eval_dist


if __name__ == '__main__':
    # create output dir
    output_dir = hparams.ROOT_PATH + 'output'
    if not os.path.exists(hparams.ROOT_PATH + 'output'):
        os.makedirs(output_dir)
        
    # load inputs
    with open(hparams.LETTER_TO_TEXT_FILE, 'rb') as f1:
        letter_to_idx = pickle.load(f1)
            
    with open(hparams.TRAIN_SEQ_FILE, 'rb') as f:
        data_seq = pickle.load(f) 
        
    seq_1d = (list(itools.chain.from_iterable(data_seq)))
    
    # train & evaluate
    encoder_trained, decoder_trained, seq_embed_trained, eval_dist = main(seq_1d, data_seq, 
                                                                          letter_to_idx, hparams.HIDDEN_SIZE)
                
    # save
    with open(hparams.SAVE_PATH + 'encoder_trained.pkl', 'wb') as f:
        pickle.dump(encoder_trained, f)
        pickle.dump(seq_embed_trained, f)
        pickle.dump(eval_dist, f)
    