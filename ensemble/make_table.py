# hiddens, dropout, embed_dims, embedders, use_pos_weights,
#                                                         single_pos_weight, epochs, true_ensemble, len(preload_dirs) > 0,
#                                                         unfreeze

import os

for dirname in os.listdir('.'):
    if not dirname.startswith('['):
        continue
    props = dirname.split('_')
    assert 10 >= len(props) >= 7, props
    if len(props) == 7:
        hiddens, dropout, embed_dims, embedders, use_pos_weights, single_pos_weight, epochs = props
        true_ensemble = 'False'
        preloaded = 'False'
        unfreeze = 'False'
    if len(props) == 8:
        hiddens, dropout, embed_dims, embedders, use_pos_weights, single_pos_weight, epochs, true_ensemble = props
        preloaded = true_ensemble
        unfreeze = 'False'
    if len(props) == 9:
        hiddens, dropout, embed_dims, embedders, use_pos_weights, single_pos_weight, epochs, true_ensemble, \
            preloaded = props
        unfreeze = 'False'
    if len(props) == 10:
        hiddens, dropout, embed_dims, embedders, use_pos_weights, single_pos_weight, epochs, true_ensemble, \
            preloaded, unfreeze = props
    hiddens = [int(h) for h in hiddens.strip('[]').split()]
    dropout = float(dropout)
    embed_dims = int(embed_dims)
    embedders = [h.strip("'") for h in hiddens.strip('[]').split()]
    use_pos_weights = bool(use_pos_weights)
    single_pos_weight = bool(single_pos_weight)
    epochs = int(epochs)
    true_ensemble = bool(true_ensemble)
    preloaded = bool(preloaded)
    unfreeze = bool(unfreeze)