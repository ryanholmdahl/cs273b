from ensemble.data_manager import DataManager


class ProteinDataManager(DataManager):
    def __init__(self, train_dbids, dev_dbids, test_dbids, use_cuda):
        super(ProteinDataManager, self).__init__(train_dbids, dev_dbids, test_dbids, use_cuda)

    def connect_to_submodule(self, submodule):
        pass

    def sample_train_batch(self, dbids):
        pass

    def sample_dev_batch(self, dbids):
        pass

    def sample_test_batch(self, dbids):
        pass
