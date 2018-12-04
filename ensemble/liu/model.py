from ensemble.liu.liu_model import LiuEmbeddingModel


def load_liu_models(n_terms, embed_size):
    return [LiuEmbeddingModel(n_terms, embed_size)]
