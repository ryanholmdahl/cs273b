from ensemble.model import EnsembleModel
from ensemble.text.model import load_text_models
from ensemble.protein.model import load_protein_models
from ensemble.go.model import load_go_models
from ensemble.data_manager import EnsembleDataManager
from ensemble.text.data_manager import TextDataManager
from ensemble.protein.data_manager import ProteinDataManager
from ensemble.go.data_manager import GoDataManager
import argparse
import torch.nn as nn
import torch.optim as optim
from src.text_model_pipeline import compute_metrics

from pytorch_classification.utils import AverageMeter, Bar


# TODO: check against test
# TODO: filter configs
# TODO: load checkpoints


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--hiddens', nargs='+', type=int)
    args = parser.parse_args()
    return args.cuda, args.hiddens


def _load_data_manager(cuda):
    return EnsembleDataManager(cuda, 800, [
        # (
        #     ProteinDataManager, [
        #         100,
        #     ],
        # ),
        (
            TextDataManager, [
                300, 50,
            ],
        ),
        # (
        #     GoDataManager, []
        # ),
    ])


def _load_submodules(data_manager):
    return (
        # load_protein_models(data_manager.submodule_managers[0].vocab.n_words) #+
        load_text_models(data_manager.submodule_managers[0].vocab.n_words) #+
        # load_go_models(data_manager.submodule_managers[2].num_terms)
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
    best_mAP_macro = 0.
    mAP_macro_test = 0.
    acc = AverageMeter()
    total_positive_labels = (
        data_manager.train_labels.sum() + data_manager.dev_labels.sum() + data_manager.test_labels.sum()
    )
    total_labels = (
        data_manager.train_labels.nelement() + data_manager.dev_labels.nelement() + data_manager.test_labels.nelement()
    )
    print(total_labels/total_positive_labels)
    criterion = nn.BCEWithLogitsLoss(pos_weight=total_labels/total_positive_labels)
    print(len(list(model.parameters())))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100):
        for i in range(0, len(data_manager.train_dbids), 128):
            train_inputs, targets = data_manager.sample_train_batch(128)
            logits = model.forward(train_inputs)
            loss = criterion(logits.reshape(-1), targets.reshape(-1))
            train_losses.update(loss.item(), 128)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            bar.suffix = '({epoch}/{max_epochs}) | TLoss: {train_loss:.3f} | DLoss: {dev_loss:.3f} ' \
                         '| Acc: {acc:.3f} | P: {p:.3f}| R: {r:.3f}| F: {f:.3f}| mAP: {mAP:.3f} ' \
                         '| dev mAP {mAP_best:.3f} | test mAP {mAP_test:.3f}' \
                .format(
                        epoch=epoch,
                        max_epochs=100,
                        train_loss=train_losses.avg,
                        dev_loss=dev_losses.avg,
                        acc=acc.avg,
                        p=p_micro.avg,
                        r=r_micro.avg,
                        f=f_micro.avg,
                        mAP=mAP_macro.avg,
                        mAP_best=best_mAP_macro,
                        mAP_test=mAP_macro_test
                )
            bar.next()

        dev_inputs, targets = data_manager.sample_dev_batch(121)
        logits = model.forward(dev_inputs)
        loss = criterion(logits.reshape(-1), targets.reshape(-1))
        dev_losses.update(loss.item(), 121)
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
        p_macro.update(batch_p_macro, 121)
        p_micro.update(batch_p_micro, 121)
        r_macro.update(batch_r_macro, 121)
        r_micro.update(batch_r_micro, 121)
        f_macro.update(batch_f_macro, 121)
        f_micro.update(batch_f_micro, 121)
        s_macro.update(batch_s_macro, 121)
        mAP_micro.update(batch_mAP_micro, 121)
        mAP_macro.update(batch_mAP_macro, 121)
        acc.update(batch_acc, 121)
        if batch_mAP_macro > best_mAP_macro:
            best_mAP_macro = batch_mAP_macro
            test_inputs, targets = data_manager.sample_test_batch(154)
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
            mAP_macro_test = batch_mAP_macro

    return model


def _main():
    cuda, hiddens = _parse_args()
    print('Loading data manager...')
    data_manager = _load_data_manager(cuda)
    print('Data manager loaded.')
    submodules = _load_submodules(data_manager)
    data_manager.connect_to_model(submodules)
    model = EnsembleModel(32 * 3, hiddens, 1121, submodules, 0.25)
    if cuda:
        model = model.cuda()
    _train(data_manager, model)


if __name__ == '__main__':
    _main()
