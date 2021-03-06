# Data
- Data is saved on VM in `/data/data_targetProteins`.

# Models
## Target protein sequences
- seq2seq autoencoder model trained to predict its own sequence
- Once trained, we can get an embedded representation from the encoder for target protein sequence feature
### Code
- Base model: `EncoderRNN` and `DecoderRNN` in `model_seq2seq.py`
- The module `seqAutoencoder` in `model_seq2seq.py` outputs last hidden layer embedding of all protein sequences of one drug
as well as the loss. If a drug is associated with multiple target proteins, the embeddings will be aggregated (default: average the embedding vectors). Detailed usage please refer to `train_seq2seq.py`.
- Run `src/main_seq2seq.py`, the pre-trained embedding will be saved in the directory `/data/save_targetProteins`
- Hyperparameters in `src/hparams_seq2seq.py` might be tuned. `HIDDEN_SIZE` is the dimension of feature vector which will be
used for the ensemble model
- No minibatch implemented, training might be slow

## Target protein GO terms (molecular functions)
- LSTM classification model trained to predict the label of the GO term (one hot encoding)
- Once trained, we can get an embedded representation from the LSTM for target protein GO feature
### Code
- Base model: `lstmClassifier` in `model_lstm.py`
- The module `gotermsClassifier` in `model_lstm.py` outputs last hidden layer embedding of all GO terms of one drug
as well as the loss. If a drug is associated with multiple target protein GO terms, the embeddings will be aggregated (default: taking the mean). Detailed usage please refer to `train_lstm.py`.
- Run `src/main_lstm_go.py`, the pre-trained embedding will be saved in `/data/save_targetProteins`
- Hyperparameters in `src/hparams_lstm.py` might be tuned. `HIDDEN_SIZE` is the dimension of feature vector which will be
used for the ensemble model
- No minibatch implemented

## Target proteins (protein IDs)
- LSTM classification model trained to predict the label of the target proteins (one hot encoding)
- Once trained, we can get an embedded representation from the LSTM for target protein
### Code
- Uses the same model as GO terms. The module `gotermsClassifier` in `model_lstm.py` outputs last hidden layer embedding of all GO terms of one drug as well as the loss. If a drug is associated with multiple target protein GO terms, the embeddings will be aggregated (default: taking the mean). Detailed usage please refer to `train_lstm.py`.
- Run `src/main_lstm_target.py`, the pre-trained embedding will be saved in `/data/save_targetProteins`
- Hyperparameters in `src/hparams_lstm.py` might be tuned. `HIDDEN_SIZE` is the dimension of feature vector which will be
used for the ensemble model
- No minibatch implemented
