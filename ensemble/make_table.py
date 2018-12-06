# hiddens, dropout, embed_dims, embedders, use_pos_weights,
#                                                         single_pos_weight, epochs, true_ensemble, len(preload_dirs) > 0,
#                                                         unfreeze

import os
import csv

entries = []
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
    hiddens = '/'.join([h for h in hiddens.strip('[]').split()])
    dropout = float(dropout)
    embed_dims = int(embed_dims)
    embedders = [h.strip("'") for h in embedders.strip('[]').split()]
    use_pos_weights = bool(use_pos_weights)
    single_pos_weight = bool(single_pos_weight)
    epochs = int(epochs)
    true_ensemble = bool(true_ensemble)
    preloaded = bool(preloaded)
    unfreeze = bool(unfreeze)
    with open(os.path.join(dirname, 'map_test.txt'), 'rt') as infile:
        mAP = int(infile.readlines()[0].strip())
    entries.append([hiddens, dropout, embed_dims, 'text' in embedders, 'protein' in embedders, 'liu' in embedders,
                    use_pos_weights, single_pos_weight, true_ensemble, preloaded, unfreeze, mAP])

with open('table.csv', 'wt') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['hiddens', 'dropout', 'embed_dim', 'uses_text', 'uses_protein', 'uses_liu', 'pos_weights',
                     'average_pos_weight', 'ensemble', 'preloaded', 'unfreeze_embeds'])
    for entry in entries:
        writer.write(entry)
