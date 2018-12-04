from torch import nn


class GoEmbeddingModel(nn.Module):
    def __init__(self, num_terms, embed_size):
        super(GoEmbeddingModel, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(num_terms, 128),
            nn.ReLU(),
            nn.Linear(128, embed_size)
        )

        self.file_name = 'go.pt'

    def forward(self, go_terms):
        return self.base(go_terms)
