import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from utils import seq2letter, prepare_seq, aggregateSeq, plotTrainValLoss
import hparams_seq2seq as hparams
from eval_seq2seq import evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed_all
np.random.seed(1)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optim,
          decoder_optim, criterion, max_length=hparams.MAX_LENGTH, teacher_enforcing=True):
    encoder.hidden = encoder.init_hidden()
    decoder.hidden = decoder.init_hidden()
    
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
    loss = 0
    
    for ei in range(input_length):
        encoder_output, last_hidden = encoder(torch.tensor([input_tensor[ei]], device=device))
        encoder_outputs[ei] = encoder_output[0,0]
    
    decoder_input = torch.tensor([[hparams.SOS_token]], device=device)
    
    decoder.hidden = encoder.hidden
      
    if teacher_enforcing: # with teacher forcing
        for di in range(target_length):
            decoder_output = decoder(decoder_input)
            loss += criterion(decoder_output, torch.tensor([target_tensor[di]], device=device))
    
            decoder_input = target_tensor[di] # teacher forcing, feed target as the next input
    else: # without teacher forcing   
        for di in range(target_length):
            decoder_output = decoder(decoder_input)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            
            loss += criterion(decoder_output, torch.tensor([target_tensor[di]], device=device))
            if decoder_input.item() == hparams.EOS_token:
                break
        
    loss.backward()
    
    encoder_optim.step()
    decoder_optim.step()
    
    return loss.item() / target_length


def trainIters(seq, letter_to_idx, encoder, decoder, n_iters):
    
    encoder_optim = optim.SGD(encoder.parameters(), lr = hparams.LEARNING_RATE, momentum = 0.9)
    decoder_optim = optim.SGD(decoder.parameters(), lr = hparams.LEARNING_RATE, momentum = 0.9)
    
    criterion = nn.NLLLoss()
    
    # split train & val sets
    indices = np.random.permutation(len(seq))
    bound = int(round(0.8*len(seq)))
    train_idx = indices[:bound]
    val_idx = indices[bound:]
    train_seq = [seq[i] for i in train_idx]
    val_seq = [seq[i] for i in val_idx]
    
    lowest_hamm_dist = 0.0
    plot_val_losses = []
    plot_train_losses = []
    for epoch in range(n_iters):
        print("Current epoch:{}".format(epoch))
        
        # train
        train_loss_total = 0.0
        for idx_targ in range(len(train_seq)): # for each target sequence
            if idx_targ % 100 == 0:
                print("idx_targ:{}".format(idx_targ))
            curr_seq = train_seq[idx_targ]
            input_seq_letters = seq2letter(curr_seq)
            input_tensor = prepare_seq(input_seq_letters, letter_to_idx)
            target_tensor = input_tensor # in our case, we will just use the input as the targets
            
            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optim,
                         decoder_optim, criterion, teacher_enforcing=hparams.TEACHER_ENFORCING)
            train_loss_total += loss
        
        avg_train_loss = train_loss_total / len(train_seq)
        plot_train_losses.append(avg_train_loss)     
        train_loss_total = 0
        
        # evaluate
        val_loss_total = 0.0
        val_dist_total = 0.0
        for idx_targ in range(len(val_seq)):
            curr_seq = val_seq[idx_targ]
            input_seq_letters = seq2letter(curr_seq)
            input_tensor = prepare_seq(input_seq_letters, letter_to_idx)      
            loss, hamming_dist = evaluate(val_seq, encoder, decoder, letter_to_idx, criterion)
            val_loss_total += loss
            val_dist_total += hamming_dist
        val_loss = val_loss_total / len(val_seq)
        val_dist = val_dist_total / len(val_seq)
        plot_val_losses.append(val_loss)
        
        if val_dist < lowest_hamm_dist:
            lowest_hamm_dist = val_dist
            os.system('rm ' + hparams.SAVE_PATH + 'bestModel_valDist*.model')
            print('New lowest validation dist!')
            torch.save(encoder.state_dict(), hparams.SAVE_PATH + 'bestModel_valDist' + str(int(lowest_hamm_dist*10000)) + '.model')
                
#    plotTrainValLoss(plot_train_losses, plot_val_losses, hparams.SAVE_PATH)

    return encoder, decoder, lowest_hamm_dist


def get_trained_embeds(train_seq, letter_to_idx, encoder):
    
    embeds_trained = []
    with torch.no_grad():
        for idx_drug in range(len(train_seq)):
            drug_seq = train_seq[idx_drug]
    
            if len(drug_seq) == 0:
                aggreg_embeds = torch.zeros(hparams.HIDDEN_SIZE, device=device)
            else:
                encoder.hidden = encoder.init_hidden()
                all_embeds = torch.zeros(0, 0, device=device)
                for idx_targ in range(len(drug_seq)):
                    curr_seq = drug_seq[idx_targ]
                    input_seq_letters = seq2letter(curr_seq)
                    input_tensor = prepare_seq(input_seq_letters, letter_to_idx)
                    for ei in range(input_tensor.size(0)):
                        output, last_hidden = encoder(torch.tensor([input_tensor[ei]], device=device))
                    all_embeds = torch.cat((all_embeds, last_hidden), dim = 0)
#                aggreg_embeds = aggregateSeq(all_embeds, 'MEAN', keep_dim=False)
                aggreg_embeds = encoder.aggregate(all_embeds, 'MEAN', keep_dim=False)
            
            embeds_trained.append(list(aggreg_embeds))
        
        embeds_trained = torch.tensor(embeds_trained, device=device)
            
    return embeds_trained



    
        
            

        
        
        
        
        
        
        
        
        
    
    
    
    
