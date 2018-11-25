from torch import nn
import torch


class EnsembleModel(nn.Module):
    def __init__(self, embed_dim, hidden_dims, output_dim, submodules, dropout):
        super(EnsembleModel, self).__init__()
        self.submodules = submodules
        fcs = []
        prev_dim = embed_dim
        for hidden_dim in hidden_dims:
            fcs.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
            prev_dim = hidden_dim
        fcs.append(nn.Linear(prev_dim, output_dim))
        self.fc = nn.Sequential(nn.Dropout(dropout), *fcs)

        for i, submodule in enumerate(self.submodules):
            self.add_module('submodule{}'.format(i), submodule)

    def forward(self, submodule_inputs):
        embeds = [submodule(*submodule_input)
                  for submodule, submodule_input in zip(self.submodules, submodule_inputs)]
        embed = torch.cat(embeds, dim=1)
        return self.fc(embed)
