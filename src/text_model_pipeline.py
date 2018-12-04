import sys
sys.path.append('../..')

import time
import torch
import sklearn
import warnings
import numpy as np

from pytorch_classification.utils import Bar, AverageMeter


def compute_metrics(logit, target):
    prob = torch.sigmoid(logit).detach().cpu().numpy()
    pred = (logit > 0).detach().int().cpu().numpy()
    target = target.cpu().numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_micro, r_micro, f_micro, s_micro \
            = sklearn.metrics.precision_recall_fscore_support(y_true=target, y_pred=pred, average='micro')
        p_macro, r_macro, f_macro, s_macro \
            = sklearn.metrics.precision_recall_fscore_support(y_true=target, y_pred=pred, average='macro')
        # mAP_macro = sklearn.metrics.average_precision_score(y_true=target, y_score=prob, average='macro')
        # aps = []
        # for se in range(len(target[0])):
        #     ap = sklearn.metrics.average_precision_score(target[:, se].reshape((-1)), prob[:, se].reshape((-1)))
        #     if ap >= 0:
        #         aps.append(ap)
        # mAP_macro = sum(aps) / len(aps)
        mAP_micro = 0.
        mAP_micro = sklearn.metrics.average_precision_score(y_true=target, y_score=prob, average='micro')

    if not s_macro:
        s_macro = 0
    if not s_micro:
        s_micro = 0

    acc = np.sum(pred == target)/float(pred.size)

    return p_micro, r_micro, f_micro, s_micro, p_macro, r_macro, f_macro, s_macro, mAP_micro, mAP_macro, acc


def train(model, dm, loss_criterion, optimizer, args):
    model.train()  # switch to train mode

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    p_micro = AverageMeter()
    r_micro = AverageMeter()
    f_micro = AverageMeter()
    p_macro = AverageMeter()
    r_macro = AverageMeter()
    f_macro = AverageMeter()
    s_macro = AverageMeter()
    mAP_micro = AverageMeter()
    mAP_macro = AverageMeter()
    acc = AverageMeter()

    end = time.time()

    bar = Bar('Processing', max=args.batches_per_epoch)
    batch_idx = 0

    while batch_idx < args.batches_per_epoch:
        # sample batch
        (des, des_unsort, ind, ind_unsort, act, act_unsort, targets) = dm.sample_train_batch(
            batch_size=args.batch_size, embed1=model.glove_embed, embed2=model.other_embed, use_cuda=args.cuda)
        encoder_init_hidden = model.encoder.initHidden(batch_size=args.batch_size)

        if args.cuda:
            model = model.cuda()
            targets = targets.cuda()
            if len(encoder_init_hidden):
                encoder_init_hidden = [
                    x.cuda() for x in encoder_init_hidden]
            else:
                encoder_init_hidden = encoder_init_hidden.cuda()
            loss_criterion = loss_criterion.cuda()

        # measure data loading timeult
        data_time.update(time.time() - end)

        # compute output
        logit_output = model(des_embed=des, des_unsort=des_unsort,
                             ind_embed=ind, ind_unsort=ind_unsort,
                             act_embed=act, act_unsort=act_unsort,
                             encoder_init_hidden=encoder_init_hidden,
                             batch_size=args.batch_size)
        loss = loss_criterion(logit_output, targets)

        # measure precision, recall, fscore, support and record loss
        batch_p_micro, batch_r_micro, batch_f_micro, batch_s_micro, batch_p_macro, batch_r_macro, batch_f_macro\
            , batch_s_macro, batch_mAP_micro, batch_mAP_macro, batch_acc = compute_metrics(logit=logit_output, target=targets)

        p_macro.update(batch_p_macro, args.batch_size)
        p_micro.update(batch_p_micro, args.batch_size)
        r_macro.update(batch_r_macro, args.batch_size)
        r_micro.update(batch_r_micro, args.batch_size)
        f_macro.update(batch_f_macro, args.batch_size)
        f_micro.update(batch_f_micro, args.batch_size)
        s_macro.update(batch_s_macro, args.batch_size)
        mAP_micro.update(batch_mAP_micro, args.batch_size)
        mAP_macro.update(batch_mAP_macro, args.batch_size)
        acc.update(batch_acc, args.batch_size)

        losses.update(loss.item(), args.batch_size)

        # compute gradient
        optimizer.zero_grad()
        loss.backward()

        # optimizer step
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        batch_idx += 1

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s' \
                     '| Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc:.3f} ' \
                     '| P: {p:.3f}| R: {r:.3f}| F: {f:.3f}| mAP mic: {mAP:.3f}|' \
            .format(
            batch=batch_idx,
            size=args.batches_per_epoch,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acc.avg,
            p=p_micro.avg,
            r=r_micro.avg,
            f=f_micro.avg,
            mAP=mAP_micro.avg,
        )
        bar.next()
    bar.finish()

    return losses.avg, acc.avg


def test(model, dm, loss_criterion, args, is_dev=True):
    model.eval()  # switch to train mode

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    p_micro = AverageMeter()
    r_micro = AverageMeter()
    f_micro = AverageMeter()
    p_macro = AverageMeter()
    r_macro = AverageMeter()
    f_macro = AverageMeter()
    s_macro = AverageMeter()
    mAP_micro = AverageMeter()
    mAP_macro = AverageMeter()
    acc = AverageMeter()

    end = time.time()

    bar = Bar('Processing', max=args.batches_per_epoch)
    batch_idx = 0

    if is_dev:
        batches_per_epoch = args.dev_batches_per_epoch
        batch_size = args.dev_batch_size
    else:
        batches_per_epoch = args.test_batches_per_epoch
        batch_size = args.test_batch_size

    while batch_idx < batches_per_epoch:
        # sample batch
        if is_dev:
            (des, des_unsort, ind, ind_unsort, act, act_unsort, targets) = dm.sample_dev_batch(
                batch_size=batch_size, embed1=model.glove_embed, embed2=model.other_embed, use_cuda=args.cuda)
        else:
            (des, des_unsort, ind, ind_unsort, act, act_unsort, targets) = dm.sample_test_batch(
                batch_size=batch_size, embed1=model.glove_embed, embed2=model.other_embed, use_cuda=args.cuda)
        encoder_init_hidden = model.encoder.initHidden(batch_size=batch_size)

        if args.cuda:
            model = model.cuda()
            targets = targets.cuda()
            if len(encoder_init_hidden):
                encoder_init_hidden = [
                    x.cuda() for x in encoder_init_hidden]
            else:
                encoder_init_hidden = encoder_init_hidden.cuda()
            loss_criterion = loss_criterion.cuda()

        # measure data loading timeult
        data_time.update(time.time() - end)

        # compute output
        logit_output = model(des_embed=des, des_unsort=des_unsort,
                             ind_embed=ind, ind_unsort=ind_unsort,
                             act_embed=act, act_unsort=act_unsort,
                             encoder_init_hidden=encoder_init_hidden,
                             batch_size=batch_size)
        loss = loss_criterion(logit_output, targets)

        # measure precision, recall, fscore, support and record loss
        batch_p_micro, batch_r_micro, batch_f_micro, batch_s_micro, batch_p_macro, batch_r_macro, batch_f_macro\
            , batch_s_macro, batch_mAP_micro, batch_mAP_macro, batch_acc = compute_metrics(logit=logit_output, target=targets)

        p_macro.update(batch_p_macro, batch_size)
        p_micro.update(batch_p_micro, batch_size)
        r_macro.update(batch_r_macro, batch_size)
        r_micro.update(batch_r_micro, batch_size)
        f_macro.update(batch_f_macro, batch_size)
        f_micro.update(batch_f_micro, batch_size)
        s_macro.update(batch_s_macro, batch_size)
        mAP_micro.update(batch_mAP_micro, batch_size)
        mAP_macro.update(batch_mAP_macro, batch_size)
        acc.update(batch_acc, batch_size)

        losses.update(loss.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        batch_idx += 1

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s' \
                     '| Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc:.3f} ' \
                     '| P: {p:.3f}| R: {r:.3f}| F: {f:.3f}| mAP mic: {mAP:.3f}| mAP mac: {mAP2:.3f}|' \
            .format(
            batch=batch_idx,
            size=args.batches_per_epoch,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acc.avg,
            p=p_micro.avg,
            r=r_micro.avg,
            f=f_micro.avg,
            mAP=mAP_micro.avg,
            mAP2=mAP_macro.avg,
        )
        bar.next()
    bar.finish()

    return losses.avg, acc.avg
