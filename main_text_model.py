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
from src.utils import dotdict, load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)

args = dotdict({
    'n_label': 1121,  # 3 classes
    'train_num_drugs': 800,
    'lr': 0.05,
    'learning_rate_decay': 0.98,
    'max_len': 300,
    'epochs': 10,
    'batch_size': 100,
    'batches_per_epoch': 50,
    'dev_batch_size': 50,
    'dev_batches_per_epoch': 10,
    'dev_batch_size': 154,
    'test_batches_per_epoch': 1,
    'hidden_size': 512,
    'lstm_layer': 1,
    'bidirectional': False,
    'glove_embedding_size': 300,
    'other_embedding_size': 200,
    'embedding_size': 500,
    'fix_emb_glove': True,
    'fix_emb_other': False,
    'dp_ratio': 0.3,
    'mlp_hidden_size_list': [256, 64],
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

    best_dev_map = 0
    best_train_map = -np.infty

    # load trained model from checkpoint
    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]
        print('loading from checkpoint in {}'.format(checkpoint_dir))
        load_checkpoint(model, checkpoint=checkpoint_dir)
        state['lr'] = 0.01
        print('resetting lr as {}'.format(state['lr']))

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        print('lr {}'.format(state['lr']))
        optimizer = optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=state['lr'])
        logger.info('\nEpoch: [{} | {}] LR: {}'.format(
            epoch + 1, args.epochs, state['lr']))

        if args.cuda:
            model.cuda()
        train_loss, train_map = pipeline.train(
            model=model,
            dm=dm,
            loss_criterion=criterion,
            optimizer=optimizer,
            args=args,
        )
        dev_loss, dev_map = pipeline.test(
            model=model,
            dm=dm,
            loss_criterion=criterion,
            args=args,
            is_dev=True
        )
        if dev_map > best_dev_map:
            print('New best model: {} vs {}'.format(dev_map, best_dev_map))
            best_dev_acc = dev_map
            save_checkpoint(
                state={
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'map': dev_map,
                    'best_acc': best_dev_acc,
                    'optimizer': optimizer.state_dict()
                }, is_best=True)
        print('Saving to checkpoint')
        save_checkpoint(
            state={
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'map': dev_map,
                'best_acc': best_dev_acc,
                'optimizer': optimizer.state_dict()
            }, is_best=False)
        if train_map - best_train_map < 0.01:
            state['lr'] *= args.learning_rate_decay
        if train_map > best_train_map:
            best_train_acc = train_map
