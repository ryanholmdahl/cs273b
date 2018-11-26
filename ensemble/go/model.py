from ensemble.go.go_model import GoEmbeddingModel


def load_go_models(n_terms):
    return [GoEmbeddingModel(n_terms)]
