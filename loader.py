from torch.utils.data.dataset import Dataset, Subset


class BrainTemplateDataset(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return [data[item] for data in self.data], self.label[item]
