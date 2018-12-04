from torch import nn


class GoEmbeddingModel(nn.Module):
    def __init__(self, num_terms):
        super(GoEmbeddingModel, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(num_terms, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, go_terms):
        return self.base(go_terms)
