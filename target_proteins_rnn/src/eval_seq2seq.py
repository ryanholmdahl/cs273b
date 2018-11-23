import torch
import hparams_seq2seq as hparams
import torch.nn as nn
from utils import seq2letter, prepare_seq
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed_all

def decode_new_seq(input_tensor, encoder, decoder, letter_to_idx, criterion, max_length=hparams.MAX_LENGTH):
     
    input_length = input_tensor.size(0)
    
    encoder.hidden = encoder.init_hidden()

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
    loss = 0
    for ei in range(input_length):
        encoder_output, last_hidden = encoder(torch.tensor([input_tensor[ei]], device=device))
        encoder_outputs[ei] = encoder_output[0,0]
    
    decoder_input = torch.tensor([[hparams.SOS_token]], device=device)
    
    decoder.hidden = encoder.hidden
    
    decoded_tensor = []
      
    for di in range(input_length):
        decoder_output = decoder(decoder_input)
        topv, topi = decoder_output.topk(1)
        decoded_tensor.append(topi.item())
        loss += criterion(decoder_output, torch.tensor([input_tensor[di]], device=device))
        decoder_input = topi.squeeze().detach()    

        if topi.item() == hparams.EOS_token:
            decoded_tensor.append('<EOS>')
            break  
    
    decoded_tensor = torch.tensor(decoded_tensor, device=device)
    
    return decoded_tensor, loss



def evaluate(val_seq, encoder, decoder, letter_to_idx, criterion):  
    
    hamming_dist = 0
    pdist = nn.PairwiseDistance(p=2)

    for idx in range(len(val_seq)):
        curr_seq = val_seq[idx]
        print(curr_seq)
        curr_letters = seq2letter(curr_seq)
        input_tensor = prepare_seq(curr_letters, letter_to_idx) 
        
        decoded_tensor, loss = decode_new_seq(input_tensor, encoder, decoder, letter_to_idx, criterion)
        decoded_tensor = decoded_tensor.float().view(1, len(decoded_tensor))
        input_tensor = input_tensor.float().view(1, len(input_tensor))
        hamming_dist += pdist(input_tensor.float(), decoded_tensor.float())
        
    avg_hamming_dist = hamming_dist / len(val_seq)
    
    return loss, avg_hamming_dist