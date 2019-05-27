import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms

from dataset_utils import Random_BalancedBatchSampler


class COXS2V(data.Dataset):
    def __init__(self, split, transform=None):
        #self.transform = transformers.get_basic_transformer()
        self.split = split
        self.transform = transform

        if split == "train":

            train_dataset = datasets.MNIST(
                root="datasets", train=True, download=True, transform=None)
            self.X = train_dataset.train_data.float() / 255.
            self.y = train_dataset.train_labels

            np.random.seed(2)
            ind = np.random.choice(len(train_dataset), 2000, replace=False)
            self.X = self.X[ind]
            self.y = self.y[ind]

        elif split == "val":

            test_dataset = datasets.MNIST(root="datasets",
                train=False, download=True, transform=None)

            self.X = test_dataset.test_data.float() / 255.
            self.y = test_dataset.test_labels

    def __getitem__(self, index):
        X, y = self.X[index].clone(), self.y[index].clone()

        X = self.transform(X[None])

        return X, y

    def __len__(self):
        """Return size of dataset."""
        return self.X.shape[0]


def get_coxs2v(split, batch_size=50):
    """Get MNIST dataset loader."""
    # image pre-processing
    # pre_process = transforms.Compose(
    #     [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    pre_process = transforms.Compose([
        #transforms.Resize((160, 160), interpolation=1),
        transforms.ToTensor()
    ])

    if split == "train":
        coxs2v_dataset = datasets.ImageFolder("/export/livia/data/lemoineh/COX-S2V/COX-S2V-Video-MTCNN160/video2",
                                          transform=pre_process)

    elif split == "val":
        coxs2v_dataset = datasets.ImageFolder("/export/livia/data/lemoineh/COX-S2V/COX-S2V-Video-MTCNN160/video1",
                                              transform=pre_process)

    people_per_batch = 20
    images_per_person = 5

    batch_sampler = Random_BalancedBatchSampler(coxs2v_dataset,
                                                people_per_batch,
                                                images_per_person,
                                                max_batches=1000)

    coxs2v_data_loader = torch.utils.data.DataLoader(coxs2v_dataset,
                                                     num_workers=4,
                                                     batch_sampler=batch_sampler,
                                                     pin_memory=False)

    return coxs2v_data_loader