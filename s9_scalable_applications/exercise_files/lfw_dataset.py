"""LFW dataloading."""
import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt


class LFWDataset(Dataset):
    """Initialize LFW dataset."""

    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        self.path_to_folder = path_to_folder
        self.transform = transform
        self.files = []
        for root, dirs, files in os.walk(path_to_folder):
            for file in files:
                self.files.append(os.path.join(root, file))
        self.len = len(self.files)

    def __len__(self):
        """Return length of dataset."""
        return self.len

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get item from dataset."""
        img_path = self.files[index]
        img = Image.open(img_path)
        img = self.transform(img)
        return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_to_folder", default="lfw-deepfunneled", type=str)
    parser.add_argument("-batch_size", default=512, type=int)
    parser.add_argument("-num_workers", default=1, type=int)
    parser.add_argument("-visualize_batch", action="store_true")
    parser.add_argument("-get_timing", action="store_true")
    parser.add_argument("-batches_to_check", default=100, type=int)

    args = parser.parse_args()

    lfw_trans = transforms.Compose([transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)), transforms.ToTensor()])

    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)

    # Define dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.visualize_batch:
        if args.visualize_batch:
            # Get one batch from the dataloader
            batch = next(iter(dataloader))

            # Visualize the batch of images
            grid = torchvision.utils.make_grid(batch)
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis('off')
            plt.show()

    if args.get_timing:
        # lets do some repetitions
        res = []
        for _ in tqdm(range(5)):
            start = time.time()
            for batch_idx, _batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)

        res = np.array(res)
        print(f"Timing: {np.mean(res)}+-{np.std(res)}")
