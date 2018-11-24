
import gzip
import os
import pickle
import urllib

import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms

from torchvision import datasets
from torch.utils.data import Dataset

class MNIST(data.Dataset):
   
    def __init__(self, split, transform=None):
      #self.transform = transformers.get_basic_transformer()
      self.split = split 
      self.transform = transform

      if split == "train":


        train_dataset = datasets.MNIST('/mnt/datasets/public/issam/MNIST',
                                train=True, download=True,
                               transform=None)
        self.X = train_dataset.train_data.float() / 255.
        self.y = train_dataset.train_labels
        
        np.random.seed(2)
        ind = np.random.choice(len(train_dataset), 2000, replace=False)
        self.X = self.X[ind]
        self.y = self.y[ind]
        
      elif split == "val":

        test_dataset = datasets.MNIST('/mnt/datasets/public/issam/MNIST', train=False, download=True,
                               transform=None)

        self.X = test_dataset.test_data.float() / 255.
        self.y = test_dataset.test_labels

    def __getitem__(self, index):
        X, y = self.X[index].clone(), self.y[index].clone()
        
        X = self.transform(X[None])

        return X, y

    def __len__(self):
        """Return size of dataset."""
        return self.X.shape[0]


def get_mnist(split, batch_size=50):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Normalize(
                                          mean=[0.5,0.5,0.5],
                                          std=[0.5,0.5,0.5])])


    # dataset and data loader
    mnist_dataset = MNIST(split=split,
                          transform=pre_process)

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=True)

    return mnist_data_loader