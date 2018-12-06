from ensemble.model import EnsembleModel
from ensemble.text.model import load_text_models
from ensemble.protein.model import load_protein_models
from ensemble.go.model import load_go_models
from ensemble.liu.model import load_liu_models
from ensemble.data_manager import EnsembleDataManager
from ensemble.text.data_manager import TextDataManager
from ensemble.protein.data_manager import ProteinDataManager
from ensemble.go.data_manager import GoDataManager
from ensemble.liu.data_manager import LiuDataManager
import argparse
import torch.nn as nn
import torch.optim as optim
import torch
from src.text_model_pipeline import compute_metrics
import os

from pytorch_classification.utils import AverageMeter, Bar


# TODO: freeze embeddings
# TODO: try removing text submodules
# TODO: true ensemble (no shared differentiability)
# TODO: run more epochs
# TODO: load checkpoints


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--hiddens', nargs='*', type=int, default=[])
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--embed_dims', type=int, default=32)
    parser.add_argument('--embedders', nargs='+')
    parser.add_argument('--use_pos_weight', action='store_true')
    parser.add_argument('--single_pos_weight', action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--true_ensemble', action='store_true')
    parser.add_argument('--preload_dirs', nargs='*', default=[])
    args = parser.parse_args()
    return args.cuda, args.hiddens, args.dropout, args.embed_dims, args.embedders, args.use_pos_weight, \
           args.single_pos_weight, args.epochs, args.true_ensemble, args.preload_dirs


def _load_data_manager(cuda, embedder_names):
    embedders = []
    if 'protein' in embedder_names:
        embedders.append((
            ProteinDataManager, [
                100,
            ]
        ))
    if 'text' in embedder_names:
        embedders.append((
            TextDataManager, [
                300, 50,
            ]
        ))
    if 'go' in embedder_names:
        embedders.append((
            GoDataManager, []
        ))
    if 'liu' in embedder_names:
        embedders.append((
            LiuDataManager, []
        ))
    return EnsembleDataManager(cuda, 700, embedders)


def _load_submodules(data_manager, embedder_names, embed_size, preload_dirs):
    manager_idx = 0
    models = []
    if 'protein' in embedder_names:
        models += load_protein_models(data_manager.submodule_managers[manager_idx].vocab.n_words, embed_size)
        manager_idx += 1
    if 'text' in embedder_names:
        models += load_text_models(data_manager.submodule_managers[manager_idx].vocab.n_words, embed_size)
        manager_idx += 1
    if 'go' in embedder_names:
        models += load_go_models(data_manager.submodule_managers[manager_idx].num_terms, embed_size)
        manager_idx += 1
    if 'liu' in embedder_names:
        models += load_liu_models(data_manager.submodule_managers[manager_idx].num_terms, embed_size)
        manager_idx += 1
    for model in models:
        for preload_dir in preload_dirs:
            for fname in os.listdir(preload_dir):
                if fname == model.file_name:
                    print('loading {}'.format(os.path.join(preload_dir, fname)))
                    model.load_state_dict(torch.load(os.path.join(preload_dir, fname)))
    return models


def _train(data_manager, model, epochs, use_pos_weight, single_pos_weight):
    bar = Bar('Processing', max=epochs*len(data_manager.train_dbids)/128)
    train_losses = AverageMeter()
    dev_losses = AverageMeter()
    p_micro = AverageMeter()
    r_micro = AverageMeter()
    f_micro = AverageMeter()
    p_macro = AverageMeter()
    r_macro = AverageMeter()
    f_macro = AverageMeter()
    s_macro = AverageMeter()
    mAP_micro = AverageMeter()
    mAP_macro = AverageMeter()
    best_mAP_micro_dev = 0.
    min_dev_loss = 1000.
    last_update_epoch = 0
    mAP_micro_test = 0.
    acc = AverageMeter()
    if use_pos_weight:
        total_positive_labels = (
            data_manager.train_labels.sum(dim=0)
        )
        total_negative_labels = (
            (1. - data_manager.train_labels).sum(dim=0)
        )
        if single_pos_weight:
            pos_weight = (total_negative_labels.sum() / total_positive_labels.sum()) * torch.ones(
                *total_negative_labels.shape, device=total_negative_labels.device)
        else:
            pos_weight = (total_negative_labels / total_positive_labels).clamp(min=0.05, max=20)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    print(len(list(model.parameters())), len([param for param in model.parameters() if param.requires_grad]))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for i in range(0, len(data_manager.train_dbids), 128):
            train_inputs, targets = data_manager.sample_train_batch(128)
            logits = model.forward(train_inputs)
            loss = criterion(logits, targets)
            train_losses.update(loss.item(), 128)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            bar.suffix = '({epoch}/{max_epochs}) | TLoss: {train_loss:.3f} | DLoss: {dev_loss:.3f} ' \
                         '| Acc: {acc:.3f} | P: {p:.3f}| R: {r:.3f}| F: {f:.3f}| davg mAP: {mAP:.3f} ' \
                         '| dbest mAP {mAP_best:.3f} | test mAP {mAP_test:.3f}' \
                .format(
                        epoch=epoch,
                        max_epochs=epochs,
                        train_loss=train_losses.avg,
                        dev_loss=dev_losses.avg,
                        acc=acc.avg,
                        p=p_micro.avg,
                        r=r_micro.avg,
                        f=f_micro.avg,
                        mAP=mAP_micro.avg,
                        mAP_best=best_mAP_micro_dev,
                        mAP_test=mAP_micro_test
                )
            bar.next()

        dev_inputs, targets = data_manager.sample_dev_batch(71)
        logits = model.forward(dev_inputs)
        loss = criterion(logits, targets)
        dev_losses.update(loss.item(), 71)
        (batch_p_micro,
         batch_r_micro,
         batch_f_micro,
         batch_s_micro,
         batch_p_macro,
         batch_r_macro,
         batch_f_macro,
         batch_s_macro,
         batch_mAP_micro,
         batch_mAP_macro,
         batch_acc) = compute_metrics(logit=logits, target=targets)
        p_macro.update(batch_p_macro, 71)
        p_micro.update(batch_p_micro, 71)
        r_macro.update(batch_r_macro, 71)
        r_micro.update(batch_r_micro, 71)
        f_macro.update(batch_f_macro, 71)
        f_micro.update(batch_f_micro, 71)
        s_macro.update(batch_s_macro, 71)
        mAP_micro.update(batch_mAP_micro, 71)
        mAP_macro.update(batch_mAP_macro, 71)
        acc.update(batch_acc, 71)
        if best_mAP_micro_dev < batch_mAP_micro:  # batch_mAP_micro > best_mAP_micro_dev:
            last_update_epoch = epoch
            min_dev_loss = dev_losses.avg
            best_mAP_micro_dev = batch_mAP_micro
            test_inputs, targets = data_manager.sample_test_batch(309)
            logits = model.forward(test_inputs)
            (batch_p_micro,
             batch_r_micro,
             batch_f_micro,
             batch_s_micro,
             batch_p_macro,
             batch_r_macro,
             batch_f_macro,
             batch_s_macro,
             batch_mAP_micro,
             batch_mAP_macro,
             batch_acc) = compute_metrics(logit=logits, target=targets)
            mAP_micro_test = batch_mAP_micro
        if epoch - last_update_epoch >= 50:
            break

    return mAP_micro_test


def _main():
    cuda, hiddens, dropout, embed_dims, embedders, use_pos_weights, single_pos_weight, epochs, true_ensemble, \
        preload_dirs = _parse_args()
    output_dir = '{}_{}_{}_{}_{}_{}_{}_{}'.format(hiddens, dropout, embed_dims, embedders, use_pos_weights,
                                                  single_pos_weight, epochs, true_ensemble)
    print('Loading data manager...')
    data_manager = _load_data_manager(cuda, embedders)
    print('Data manager loaded.')
    submodules = _load_submodules(data_manager, embedders, embed_dims, preload_dirs)
    data_manager.connect_to_model(submodules)
    if os.path.exists(output_dir):
        if all([os.path.exists(os.path.join(output_dir, submodule.file_name)) for submodule in submodules]):
            print('Already saved. Terminating')
    model = EnsembleModel(embed_dims * (len(embedders) + (2 if 'text' in embedders else 0)), hiddens, 5579, submodules,
                          dropout, true_ensemble)
    if cuda:
        model = model.cuda()
    mAP_test = _train(data_manager, model, epochs, use_pos_weights, single_pos_weight)
    for submodule in submodules:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        torch.save(submodule.state_dict(), os.path.join(output_dir, submodule.file_name))
    with open(os.path.join(output_dir, 'map_test.txt'), 'wt') as outfile:
        print(mAP_test, file=outfile)


if __name__ == '__main__':
    _main()
