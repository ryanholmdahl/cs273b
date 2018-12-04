from ensemble.go.go_model import GoEmbeddingModel


def load_go_models(n_terms, embed_size):
    return [GoEmbeddingModel(n_terms, embed_size)]
