import sys
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import logging

import src.dataman.text_datamanager as datamanager
import src.text_model_pipeline as pipeline
import src.model.text_model as text_model
import src.constant
from src.utils import dotdict, load_checkpoint, save_checkpoint, change_learning_rate

logger = logging.getLogger(__name__)

args = dotdict({
    'n_label': 1121,  # 3 classes
    'train_num_drugs': 800,
    'lr': 0.001,
    'learning_rate_decay': 0.9,
    'weight_decay': 5e-5,
    'balance_loss': True,
    'max_len': 300,
    'epochs': 10,
    'batch_size': 500,
    'batches_per_epoch': 10,
    'dev_batch_size': 121,
    'dev_batches_per_epoch': 1,
    'test_batch_size': 154,
    'test_batches_per_epoch': 1,
    'hidden_size': 32,
    'lstm_layer': 1,
    'bidirectional': False,
    'glove_embedding_size': 300,
    'other_embedding_size': 200,
    'embedding_size': 500,
    'fix_emb_glove': True,
    'fix_emb_other': True,
    'dp_ratio': 0.2,
    'mlp_hidden_size_list': [32, 32],
    'cuda': torch.cuda.is_available(),
})
state = {k: v for k, v in args.items()}


if __name__ == "__main__":
    print(args)

    dm = datamanager.TextDataManager(args)
    args.n_embed = dm.vocab.n_words
    model = text_model.TextClassifier(config=args)

    model.glove_embed.weight.data = torch.Tensor(dm.vocab.embed_vectors)

    # Numbers of parameters
    print("number of trainable parameters found {}".format(sum(
        param.nelement() for param in model.parameters()
        if param.requires_grad)))

    pos_weight = torch.sum(1-dm.train_labels, dim=0)/torch.sum(dm.train_labels, dim=0)

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
        checkpoint_dir = sys.argv[1]
        print('loading from checkpoint in {}'.format(checkpoint_dir))
        checkpoint = load_checkpoint(checkpoint=checkpoint_dir)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        state['lr'] = checkpoint['lr']
        best_dev_acc = checkpoint['acc']

    for epoch in range(args.epochs):
        print('lr {}'.format(state['lr']))
        logger.info('\nEpoch: [{} | {}] LR: {}'.format(
            epoch + 1, args.epochs, state['lr']))

        if args.cuda:
            model.cuda()
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
