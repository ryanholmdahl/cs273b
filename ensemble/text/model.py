from src.model.text_model import TextEmbeddingModel
from src.utils import dotdict
import torch


CONFIG = dotdict({
        'n_label': 1121,  # 3 classes
        'train_num_drugs': 800,
        'lr': 0.001,
        'learning_rate_decay': 0.9,
        'weight_decay': 5e-4,
        'balance_loss': False,
        'max_len': 300,
        'epochs': 50,
        'batch_size': 100,
        'batches_per_epoch': 100,
        'dev_batch_size': 121,
        'dev_batches_per_epoch': 1,
        'test_batch_size': 154,
        'test_batches_per_epoch': 1,
        'hidden_size': 128,  #
        'lstm_layer': 1,  #
        'bidirectional': False,  #
        'glove_embedding_size': 50,  #
        'other_embedding_size': 200,  #
        'embedding_size': 50+200,  #
        'fix_emb_glove': True,  #
        'fix_emb_other': True,  #
        'dp_ratio': 0.3,  #
        'cuda': torch.cuda.is_available(),
    })


def load_text_models(n_words, embed_size, unfreeze):
    CONFIG.n_embed = n_words
    CONFIG.embedding_size = CONFIG.glove_embedding_size + CONFIG.other_embedding_size
    CONFIG.hidden_size = embed_size
    CONFIG.fix_emb_glove = not unfreeze
    CONFIG.fix_emb_other = not unfreeze
    return [TextEmbeddingModel(CONFIG)]
