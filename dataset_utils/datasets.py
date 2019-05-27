import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from random import sample
import multiprocessing as mp

class Random_BalancedBatchSampler(BatchSampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source: datasets.ImageFolder, num_classes_per_batch: int, samples_per_class: int,
                 max_batches: int=None):

        self.data_source = data_source
        self.num_classes_per_batch = num_classes_per_batch
        self.samples_per_class = samples_per_class
        self.max_batches = max_batches

        if self.num_classes_per_batch > len(self.data_source.classes):
            raise ValueError('Trying to sample {} classes in a dataset with {} classes.'.format(
                self.num_classes_per_batch, len(self.data_source.classes)))

        self.num_batches = len(self.data_source.samples) // (self.num_classes_per_batch * self.samples_per_class)

        self.sample_idxs = np.arange(len(self.data_source.samples))
        self.targets = np.array(self.data_source.targets)
        self.classes = np.array(list(self.data_source.class_to_idx.values()))
        self.class_samples = {i: self.sample_idxs[self.targets == i] for i in self.classes}

    def __iter__(self):
        batches = []
        for i in range(min(self.num_batches, self.max_batches)):
            batch = []
            chosen_classes_idx = np.random.choice(self.classes, self.num_classes_per_batch, replace=False)
            for i in chosen_classes_idx:
                batch.append(np.random.choice(self.class_samples[i], self.samples_per_class, replace=False))
            batches.append(np.concatenate(batch))

        return iter(batches)

    def __len__(self):
        return min(self.num_batches, self.max_batches)


class PairsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
                 data_source: str,
                 pair_file: str,
                 transform: transforms,
                 preload: bool=False):
        self.data_source = data_source
        self.pair_file = pair_file
        self.transform = transform

        self.pairs, self.issame = Get_Pairs(pair_file, data_source)

        self.preloaded = False
        if preload:
            self.images = {}
            uniques = np.unique(np.array(self.pairs))
            for path in uniques:
                img = Image.open(path)
                self.images[path] = img.copy()
            self.preloaded = True

            # def load_image(fname):
            #     image = Image.open(fname)
            #     return (fname, image.copy())
            # uniques = np.unique(np.array(self.pairs))
            # with mp.Pool(4) as p:
            #     self.images = p.map(load_image, uniques)
            # self.images = {k:v for k,v in self.images}
            # self.preloaded = True


    def __len__(self):
        return len(self.issame)

    def __getitem__(self, idx):
        if self.preloaded:
            img1 = self.images[self.pairs[idx][0]]
            img2 = self.images[self.pairs[idx][1]]
        else:
            img1 = default_loader(self.pairs[idx][0])
            img2 = default_loader(self.pairs[idx][1])

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return [img1, img2], self.issame[idx], [self.pairs[idx][0], self.pairs[idx][1]]


class PairsDatasetS2V(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
                 still_source: str,
                 video_source: str,
                 pair_file: str,
                 transform: transforms,
                 fold_list: list,
                 num_folds: int=10,
                 preload: bool=True):



        self.still_source = still_source
        self.video_source = video_source
        self.pair_file = pair_file
        self.transform = transform

        self.subject_list, self.nb_folds = self._get_subject_list()
        self.nb_subject = len(self.subject_list)
        self.nb_subject_per_fold = self.nb_subject // self.nb_folds
        if num_folds != self.nb_folds:
            raise Exception('There are {} folds in pair file. Marked {} folds.'.format(self.nb_folds, num_folds))
        if max(fold_list) > self.nb_folds:
            raise Exception('Fold number {} is out of range. Maximum number of fold is {}.'.format(max(fold_list), self.nb_folds))

        self.pairs, self.issame = self._get_pairs_from_fold(fold_list)

        self.preloaded = False
        if preload:
            self.images = {}
            uniques = np.unique(np.array(self.pairs))
            for path in uniques:
                img = Image.open(path)
                self.images[path] = img.copy()
            self.preloaded = True

            # def load_image(fname):
            #     image = Image.open(fname)
            #     return (fname, image.copy())
            # uniques = np.unique(np.array(self.pairs))
            # with mp.Pool(4) as p:
            #     self.images = p.map(load_image, uniques)
            # self.images = {k:v for k,v in self.images}
            # self.preloaded = True

    def __len__(self):
        return len(self.issame)

    def __getitem__(self, idx):
        if self.preloaded:
            img1 = self.images[self.pairs[idx][0]]
            img2 = self.images[self.pairs[idx][1]]
        else:
            img1 = default_loader(self.pairs[idx][0])
            img2 = default_loader(self.pairs[idx][1])

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return [img1, img2], self.issame[idx], [self.pairs[idx][0], self.pairs[idx][1]]

    # Set up for evaluation
    def _get_pairs_from_fold(self,
                  fold_list):
        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []

        fold_subject_list = self._extract_fold_list(fold_list)

        pairs = []
        with open(self.pair_file, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)

        for pair in pairs:
            if pair[0] in fold_subject_list:
                if len(pair) == 3:

                    path0 = add_extension(
                        os.path.join(self.still_source, pair[0] + '_' + '%04d' % int(pair[1])))
                    path1 = add_extension(
                        os.path.join(self.video_source, pair[0], pair[0] + '_' + '%d' % int(pair[2])))
                    issame = True

                    if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                        path_list.append([path0, path1])
                        issame_list.append(issame)
                    else:
                        nrof_skipped_pairs += 1

                elif len(pair) == 4:

                    path0 = add_extension(
                        os.path.join(self.still_source, pair[0] + '_' + '%04d' % int(pair[1])))
                    path1 = add_extension(
                        os.path.join(self.video_source, pair[2], pair[2] + '_' + '%d' % int(pair[3])))
                    issame = False

                    if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                        path_list.append([path0, path1])
                        issame_list.append(issame)
                    else:
                        nrof_skipped_pairs += 1

        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list, issame_list

    def _get_subject_list(self):

        subject_list = []

        with open(self.pair_file, 'r') as f:

            nb_fold = f.readline().split('\t')[0]

            for line in f.readlines()[1:]:
                pair = line.strip().split()

                if len(pair) == 3:
                    if pair[0] not in subject_list:
                        subject_list.append(pair[0])

        return subject_list, int(nb_fold)

    def _extract_fold_list(self, fold_list):

        list = []
        for fold in fold_list:
            upper_idx = fold * self.nb_subject_per_fold + self.nb_subject_per_fold
            lower_idx = fold * self.nb_subject_per_fold
            list += self.subject_list[lower_idx: upper_idx]

        return list

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


def add_extension(path):
    if os.path.exists(path + '.jpg'):
        return path + '.jpg'
    if os.path.exists(path + '.JPG'):
        return path + '.JPG'
    elif os.path.exists(path + '.png'):
        return path + '.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


# Set up for evaluation
def Get_Pairs(pairs_path,
              images_path):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []

    pairs = read_pairs(pairs_path)

    path0 = ''
    path1 = ''
    issame = False

    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(images_path, pair[0], pair[1]))
            path1 = add_extension(os.path.join(images_path, pair[0], pair[2]))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(images_path, pair[0], pair[1]))
            path1 = add_extension(os.path.join(images_path, pair[2], pair[3]))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list.append([path0, path1])
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list
