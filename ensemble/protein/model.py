from src.utils import dotdict
from ensemble.protein.protein_model import ProteinEmbeddingModel
import torch

CONFIG = dotdict({
        'n_label': 1121,  # 3 classes
        'train_num_drugs': 800,
        'lr': 0.001,
        'learning_rate_decay': 0.9,
        'weight_decay': 5e-4,
        'balance_loss': False,
        'max_len': 100,
        'epochs': 50,
        'batch_size': 100,
        'batches_per_epoch': 100,
        'dev_batch_size': 121,
        'dev_batches_per_epoch': 1,
        'test_batch_size': 154,
        'test_batches_per_epoch': 1,
        'hidden_size': 64,  #
        'lstm_layer': 1,  #
        'bidirectional': False,  #
        'embedding_size': 50,  #
        'fix_emb_glove': True,  #
        'fix_emb_other': True,  #
        'dp_ratio': 0.3,  #
        'cuda': torch.cuda.is_available(),
    })


def load_protein_models(n_words):
    CONFIG.n_embed = n_words
    print(n_words)
    return [ProteinEmbeddingModel(CONFIG)]
