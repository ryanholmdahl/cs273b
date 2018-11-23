# Data
- Create a directory `data` in `target_proteins_rnn` and unzip the files in `data_targetProteins.zip` in this directory.

# Models
## Target protein sequences
- seq2seq autoencoder model trained to predict its own sequence
- Once trained, we can get an embedded representation from the encoder for target protein sequence feature
### Code
- Run `src/main_seq2seq.py`, the pre-trained embedding & best model (defined by pairwise distance between predicted sequence & actual
  on validation set) will be saved in the directory `./output`
- Hyperparameters in `src/hparams_seq2seq.py` might be tuned. `HIDDEN_SIZE` is the dimension of feature vector which will be
used for the ensemble model
- No minibatch implemented, training might be slow

## Target protein GO terms (molecular functions)
- LSTM classification model trained to predict the label of the GO term (one hot encoding)
- Once trained, we can get an embedded representation from the LSTM for target protein GO feature
### Code
- Run `src/main_lstm_go.py`, the pre-trained embedding & best model (defined by prediction accuracy) will be saved in `./output`
- Hyperparameters in `src/hparams_lstm.py` might be tuned. `HIDDEN_SIZE` is the dimension of feature vector which will be
used for the ensemble model
- No minibatch implemented

## Target proteins (protein IDs)
- LSTM classification model trained to predict the label of the target proteins (one hot encoding)
- Once trained, we can get an embedded representation from the LSTM for target protein
### Code
- Run `src/main_lstm_target.py`, the pre-trained embedding & best model (defined by prediction accuracy) will be saved in `./output`
- Hyperparameters in `src/hparams_lstm.py` might be tuned. `HIDDEN_SIZE` is the dimension of feature vector which will be
used for the ensemble model
- No minibatch implemented
