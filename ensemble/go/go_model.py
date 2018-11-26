from torch import nn


class GoEmbeddingModel(nn.Module):
    def __init__(self, num_terms):
        super(GoEmbeddingModel, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(num_terms, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

    def forward(self, go_terms):
        return self.base(go_terms)
