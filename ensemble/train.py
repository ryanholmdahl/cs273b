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
from src.text_model_pipeline import compute_metrics

from pytorch_classification.utils import AverageMeter, Bar


# TODO: freeze embeddings
# TODO: try removing text submodules
# TODO: true ensemble (no shared differentiability)
# TODO: run more epochs
# TODO: load checkpoints


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--hiddens', nargs='+', type=int)
    args = parser.parse_args()
    return args.cuda, args.hiddens


def _load_data_manager(cuda):
    return EnsembleDataManager(cuda, 700, [
        # (
        #     ProteinDataManager, [
        #         100,
        #     ],
        # ),
        # (
        #     TextDataManager, [
        #         300, 50,
        #     ],
        # ),
        # (
        #     GoDataManager, []
        # ),
        (
            LiuDataManager, []
        )
    ])


def _load_submodules(data_manager):
    return (
        # load_protein_models(data_manager.submodule_managers[0].vocab.n_words) +
        # load_text_models(data_manager.submodule_managers[1].vocab.n_words) +
        # load_go_models(data_manager.submodule_managers[2].num_terms)
        load_liu_models(data_manager.submodule_managers[0].num_terms)
    )


def _train(data_manager, model):
    bar = Bar('Processing', max=100*len(data_manager.train_dbids)/128)
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
    mAP_micro_test = 0.
    acc = AverageMeter()
    total_positive_labels = (
        data_manager.train_labels.sum(dim=0)
    )
    print(total_positive_labels.shape)
    total_negative_labels = (
        (1. - data_manager.train_labels).sum(dim=0)
    )
    print(total_negative_labels.shape)
    criterion = nn.BCEWithLogitsLoss()
    print(len(list(model.parameters())))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100):
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
                        max_epochs=100,
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
        if min_dev_loss > dev_losses.avg:  # batch_mAP_micro > best_mAP_micro_dev:
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

    return model


def _main():
    cuda, hiddens = _parse_args()
    print('Loading data manager...')
    data_manager = _load_data_manager(cuda)
    print('Data manager loaded.')
    submodules = _load_submodules(data_manager)
    data_manager.connect_to_model(submodules)
    model = EnsembleModel(64, hiddens, 5579, submodules, 0.5)
    if cuda:
        model = model.cuda()
    _train(data_manager, model)


if __name__ == '__main__':
    _main()
