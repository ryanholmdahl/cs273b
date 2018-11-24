import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from model_seq2seq import seqAutoencoder
import hparams_seq2seq as hparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed_all
np.random.seed(1)


def trainIter(train_sequences, letter_to_idx, n_epochs, save_path):
    criterion = nn.NLLLoss()
    config = {'vocab_size': len(letter_to_idx),
              'hidden_size': hparams.HIDDEN_SIZE,
              'output_size': len(letter_to_idx),
              'max_length': hparams.MAX_LENGTH,
              'teacher_enforcing': hparams.TEACHER_ENFORCING
              }
                         
    model = seqAutoencoder(config, criterion)
    optimizer = optim.SGD(model.parameters(), lr = hparams.LEARNING_RATE, momentum = 0.9)

    for epoch in range(n_epochs):  
        model.train()
        print("Current epoch:{}".format(epoch))
        # train
        all_embeds = []
        train_losses = 0.0
        for dbid, drug_seq in train_sequences.items():
            if len(drug_seq) == 0 or (len(drug_seq[0]) == 0):
                curr_embeds = torch.zeros(hparams.HIDDEN_SIZE, device=device) # if no target for this drug, output zero embeddings
            else:
                curr_embeds, loss = model(drug_seq, letter_to_idx)                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
                train_losses += loss
                
            all_embeds.append(curr_embeds)

        avg_train_loss = train_losses / len(train_sequences)
        all_embeds = torch.stack(all_embeds)

        print("Training loss in current epoch:{}".format(avg_train_loss))

    return model, all_embeds
