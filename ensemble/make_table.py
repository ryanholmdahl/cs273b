# hiddens, dropout, embed_dims, embedders, use_pos_weights,
#                                                         single_pos_weight, epochs, true_ensemble, len(preload_dirs) > 0,
#                                                         unfreeze

import os
import csv

best_by_embedder = {
    'liu': (0., None),
    'protein': (0., None),
    'text': (0., None),
}

entries = []
for dirname in os.listdir('.'):
    if not dirname.startswith('['):
        continue
    print(dirname)
    props = dirname.split('_')
    assert 11 == len(props), props
    hiddens, dropout, embed_dims, embedders, use_pos_weights, single_pos_weight, epochs, true_ensemble, \
        preloaded, unfreeze, lr = props
    hiddens = '/'.join([h for h in hiddens.strip('[]').split()])
    dropout = float(dropout)
    embed_dims = int(embed_dims)
    embedders = [h.strip("'") for h in embedders.strip('[]').split()]
    assert len(embedders) == 1
    use_pos_weights = bool(use_pos_weights)
    single_pos_weight = bool(single_pos_weight)
    epochs = int(epochs)
    true_ensemble = bool(true_ensemble)
    preloaded = bool(preloaded)
    unfreeze = bool(unfreeze)
    lr = float(lr)
    with open(os.path.join(dirname, 'map_test.txt'), 'rt') as infile:
        mAP_test = float(infile.readlines()[0].strip())
    with open(os.path.join(dirname, 'map_dev.txt'), 'rt') as infile:
        mAP_dev = float(infile.readlines()[0].strip())
    if mAP_dev > best_by_embedder[embedders[0]][0]:
        best_by_embedder[embedders[0]] = (mAP_dev, dirname)
    entries.append([hiddens, dropout, embed_dims, 'text' in embedders, 'protein' in embedders, 'liu' in embedders,
                    use_pos_weights, single_pos_weight, true_ensemble, preloaded, unfreeze, lr, mAP_dev, mAP_test])

print(best_by_embedder)

with open('table.csv', 'wt') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['hiddens', 'dropout', 'embed_dim', 'uses_text', 'uses_protein', 'uses_liu', 'pos_weights',
                     'average_pos_weight', 'ensemble', 'preloaded', 'unfreeze_embeds'])
    for entry in entries:
        writer.writerow(entry)
