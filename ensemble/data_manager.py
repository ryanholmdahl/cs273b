import ensemble.constant as constant
import pickle
import torch
import random


class DataManager:
    def __init__(self, train_dbids, dev_dbids, test_dbids, use_cuda):
        self.train_dbids = train_dbids
        self.dev_dbids = dev_dbids
        self.test_dbids = test_dbids
        self.use_cuda = use_cuda

    def connect_to_submodule(self, submodule):
        raise NotImplementedError

    def sample_train_batch(self, dbids):
        raise NotImplementedError

    def sample_dev_batch(self, dbids):
        raise NotImplementedError

    def sample_test_batch(self, dbids):
        raise NotImplementedError


class EnsembleDataManager:
    def __init__(self, cuda, train_num_drugs, submodule_managers):
        # get train, dev, and test DBIDs using args
        with open(constant.TRAIN_IDS, "r") as fd:
            lines = fd.readlines()
            dbids = [line.strip('\n') for line in lines]
            self.train_dbids = dbids[0:train_num_drugs]
            self.dev_dbids = dbids[train_num_drugs:]
        with open(constant.TEST_IDS, "r") as fd:
            lines = fd.readlines()
            self.test_dbids = [line.strip('\n') for line in lines]

        with open(constant.TRAIN_LABELS, "rb") as fd:
            train_label_dict = pickle.load(fd)
        with open(constant.TEST_LABELS, "rb") as fd:
            test_label_dict = pickle.load(fd)

        self.train_labels = torch.Tensor([train_label_dict[dbid] for dbid in self.train_dbids])
        self.dev_labels = torch.Tensor([train_label_dict[dbid] for dbid in self.dev_dbids])
        self.test_labels = torch.Tensor([test_label_dict[dbid] for dbid in self.test_dbids])

        self.submodule_managers = []
        for submodule_manager, args in submodule_managers:
            self.submodule_managers.append(submodule_manager(self.train_dbids, self.dev_dbids, self.test_dbids, cuda,
                                                             *args))

    def connect_to_model(self, submodules):
        for submodule_manager, submodule in zip(self.submodule_managers, submodules):
            submodule_manager.connect_to_submodule(submodule)

    def sample_train_batch(self, batch_size):
        dbids = random.sample(self.train_dbids, batch_size)
        return [submodule_manager.sample_train_batch(dbids) for submodule_manager in self.submodule_managers]

    def sample_dev_batch(self, batch_size):
        dbids = random.sample(self.dev_dbids, batch_size)
        return [submodule_manager.sample_dev_batch(dbids) for submodule_manager in self.submodule_managers]

    def sample_test_batch(self, batch_size):
        dbids = random.sample(self.test_dbids, batch_size)
        return [submodule_manager.sample_test_batch(dbids) for submodule_manager in self.submodule_managers]