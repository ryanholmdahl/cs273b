import sys
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import logging

import src.dataman.text_datamanager as datamanager
import src.text_model_pipeline as pipeline
import src.model.text_model as text_model
import src.constant as constant
from src.utils import dotdict, load_checkpoint, save_checkpoint, change_learning_rate

logger = logging.getLogger(__name__)

args = dotdict({
    'n_label': 1121,  # 3 classes
    'train_num_drugs': 800,
    'lr': 0.001,
    'learning_rate_decay': 0.95,
    'weight_decay': 5e-4,
    'balance_loss': True,
    'max_len': 300,
    'epochs': 50,
    'batch_size': 100,
    'batches_per_epoch': 100,
    'dev_batch_size': 121,
    'dev_batches_per_epoch': 1,
    'test_batch_size': 154,
    'test_batches_per_epoch': 1,
    'hidden_size': 10,
    'lstm_layer': 1,
    'bidirectional': False,
    'glove_embedding_size': 50,
    'other_embedding_size': 200,
    'embedding_size': 50+200,
    'fix_emb_glove': True,
    'fix_emb_other': True,
    'dp_ratio': 0.3,
    'mlp_hidden_size_list': [32, 16],
    'cuda': torch.cuda.is_available(),
})

if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]
        print('loading from checkpoint in {}'.format(constant.SAVE_DIR+checkpoint_dir))
        checkpoint = load_checkpoint(checkpoint=checkpoint_dir)
        args = checkpoint['args']

    args.embedding_size = args.glove_embedding_size + args.other_embedding_size
    state = {k: v for k, v in args.items()}
    print(args)

    dm = datamanager.TextDataManager(args)
    args.n_embed = dm.vocab.n_words
    model = text_model.TextClassifier(config=args)

    model.glove_embed.weight.data = torch.Tensor(dm.vocab.get_glove_embed_vectors())
    model.other_embed.weight.data = torch.Tensor(dm.vocab.get_medw2v_embed_vectors())

    if args.cuda:
        model.cuda()

    # Numbers of parameters
    print("number of trainable parameters found {}".format(sum(
        param.nelement() for param in model.parameters()
        if param.requires_grad)))

    pos_weight = torch.sum(1-dm.train_labels, dim=0)/torch.sum(dm.train_labels, dim=0)
    pos_weight = torch.clamp(pos_weight, min=0.1, max=10)
    if state['balance_loss']:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=state['lr'], weight_decay=state['weight_decay'])

    best_dev_acc = 0
    best_train_acc = -np.infty

    # load trained model from checkpoint
    if len(sys.argv) > 1:
        model.load_state_dict(checkpoint['state_dict'])
        if args.cuda:
            model.cuda()
            optimizer = optim.Adam(
                [param for param in model.parameters() if param.requires_grad],
                lr=state['lr'], weight_decay=state['weight_decay'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        state['lr'] = checkpoint['lr']
        best_dev_acc = checkpoint['acc']

    for epoch in range(args.epochs):
        print('lr {}'.format(state['lr']))
        logger.info('\nEpoch: [{} | {}] LR: {}'.format(
            epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = pipeline.train(
            model=model,
            dm=dm,
            loss_criterion=criterion,
            optimizer=optimizer,
            args=args,
        )
        dev_loss, dev_acc = pipeline.test(
            model=model,
            dm=dm,
            loss_criterion=criterion,
            args=args,
            is_dev=True
        )
        if dev_acc > best_dev_acc:
            print('New best model: {} vs {}'.format(dev_acc, best_dev_acc))
            best_dev_acc = dev_acc
            save_checkpoint(
                state={
                    'args': args,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': dev_acc,
                    'best_acc': best_dev_acc,
                    'optimizer': optimizer.state_dict(),
                    'lr': state['lr']
                }, is_best=True)
        print('Saving to checkpoint')
        save_checkpoint(
            state={
                'args': args,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': dev_acc,
                'best_acc': best_dev_acc,
                'optimizer': optimizer.state_dict(),
                'lr': state['lr']
            }, is_best=False)
        if train_acc - best_train_acc < 0.01:
            state['lr'] *= args.learning_rate_decay
            change_learning_rate(optimizer, state['lr'])
        if train_acc > best_train_acc:
            best_train_acc = train_acc
