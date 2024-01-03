import torch
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder

    def __len__(self):
        return 5*5000

    def __getitem__(self, idx):
        file_idx = idx // 5000
        image_idx = idx % 5000
        images = torch.load(self.folder + f"train_images_{file_idx}.pt")
        labels = torch.load(self.folder + f"train_target_{file_idx}.pt")
        return images[image_idx, :].flatten(), labels[image_idx]

class TestingDataset(Dataset):
    def __init__(self, folder):
        self.images = torch.load(folder + "test_images.pt")
        self.labels = torch.load(folder + "test_target.pt")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx, :].flatten(), self.labels[idx]


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    folder = "../../../data/corruptmnist/"
    train = torch.utils.data.DataLoader(TrainingDataset(folder), batch_size=64, shuffle=True)
    test = torch.utils.data.DataLoader(TestingDataset(folder), batch_size=64, shuffle=True)
    return train, test
