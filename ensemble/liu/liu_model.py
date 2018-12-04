from torch import nn


class LiuEmbeddingModel(nn.Module):
    def __init__(self, num_terms):
        super(LiuEmbeddingModel, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(num_terms, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, go_terms):
        return go_terms
        # return self.base(go_terms)
