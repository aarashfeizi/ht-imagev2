from datasets.base_dataset import BaseDataset


class HotelsDataset(BaseDataset):
    def __init__(self, args, mode, filename='', transform=None):
        super(HotelsDataset, self).__init__(args, mode, filename, transform)

        self.data_dict = self.make_data_dict()

        self.lbl2idx = {l: i for i, l in enumerate(self.data_dict.keys())}

        self.rename_labels()

        self.labels = list(self.data_dict.keys())