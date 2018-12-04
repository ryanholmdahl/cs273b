from torch import nn


class LiuEmbeddingModel(nn.Module):
    def __init__(self, num_terms, embed_size):
        super(LiuEmbeddingModel, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(num_terms, 128),
            # nn.BatchNorm1d(128),
            nn.Sigmoid(),
            nn.Linear(128, embed_size)
        )

        self.file_name = 'liu.pt'

    def forward(self, go_terms):
        # return go_terms
        return self.base(go_terms)
