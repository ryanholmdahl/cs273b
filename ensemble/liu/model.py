from ensemble.liu.liu_model import LiuEmbeddingModel


def load_liu_models(n_terms):
    return [LiuEmbeddingModel(n_terms)]
