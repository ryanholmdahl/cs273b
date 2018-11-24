from ensemble.model import EnsembleModel
from ensemble.text.model import load_text_models
from ensemble.data_manager import EnsembleDataManager
from ensemble.text.data_manager import TextDataManager
import argparse


# TODO: fix rel paths
# TODO: load checkpoints


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--hiddens', nargs='+', type=int)
    args = parser.parse_args()
    return args.cuda, args.hiddens


def _load_data_manager(cuda):
    return EnsembleDataManager(cuda, 800, [
        (
            TextDataManager, [
                300, 50,
            ]
        ),
    ])


def _load_submodules(data_manager):
    return load_text_models(data_manager.submodule_managers[0].vocab.n_words)


def _train(data_manager, model):
    for i in range(10):
        train_inputs = data_manager.sample_train_batch(64)
        class_scores = model.forward(train_inputs)
        print(class_scores)
    return model


def _main():
    cuda, hiddens = _parse_args()
    print('Loading data manager...')
    data_manager = _load_data_manager(cuda)
    print('Data manager loaded.')
    submodules = _load_submodules(data_manager)
    data_manager.connect_to_model(submodules)
    model = EnsembleModel(32 * 3, hiddens, 1121, submodules)
    if cuda:
        model = model.cuda()
    _train(data_manager, model)


if __name__ == '__main__':
    _main()
