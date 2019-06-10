
import os
import numpy as np
import tqdm

from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.datasets.folder import default_loader

import utils


class PairsDataset(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
                 data_source: str,
                 pair_file: str,
                 transform: transforms,
                 preload: bool=False):
        self.data_source = data_source
        self.pair_file = pair_file
        self.transform = transform

        self.pairs, self.issame = utils.Get_Pairs(pair_file, data_source)

        self.preloaded = False
        if preload:
            print('Preload images')
            self.images = {}
            uniques = np.unique(np.array(self.pairs))
            tbar = tqdm.tqdm(uniques)
            for path in tbar:
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

        return [img1, img2], self.issame[idx]


class PairsDatasetS2V(data.Dataset):
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

        self.subject_list, self.nb_folds = utils.get_subject_list(pair_file)
        self.nb_subject = len(self.subject_list)
        self.nb_subject_per_fold = self.nb_subject // self.nb_folds
        if num_folds != self.nb_folds:
            raise Exception('There are {} folds in pair file. Marked {} folds.'.format(self.nb_folds, num_folds))
        if max(fold_list) > self.nb_folds:
            raise Exception('Fold number {} is out of range. Maximum number of fold is {}.'.format(max(fold_list), self.nb_folds))

        self.pairs, self.issame = utils.get_pairs_from_fold(still_source,
                                                            video_source,
                                                            pair_file,
                                                            self.subject_list)

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

        return [img1, img2], self.issame[idx]


class DatasetS2V(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
                 still_source: str,
                 video_source: str,
                 pair_file: str,
                 transform: transforms,
                 fold_list: list,
                 num_folds: int=10,
                 preload: bool = False,
                 video_only: bool = False,
                 samples_division_list = None,  # [0.4, 0.6]
                 div_idx: int = -1):

        self.still_source = still_source
        self.video_source = video_source
        self.pair_file = pair_file
        self.transform = transform
        self.video_only = video_only

        self.subject_list, self.nb_folds = utils.get_subject_list(pair_file)
        self.nb_subject = len(self.subject_list)
        self.nb_subject_per_fold = self.nb_subject // self.nb_folds
        if num_folds != self.nb_folds:
            raise Exception('There are {} folds in pair file. Marked {} folds.'.format(self.nb_folds, num_folds))
        if max(fold_list) > self.nb_folds:
            raise Exception('Fold number {} is out of range. Maximum number of fold is {}.'.format(max(fold_list), self.nb_folds))

        # Divide samples within subject subdirectories
        if samples_division_list:
            if sum(samples_division_list) != 1.0:
                raise Exception('The sample division list is incorrect as the division sum should be 1. The sum is {}.'.format(sum(samples_division_list)))
            if (div_idx >= len(samples_division_list)) or (div_idx < 0):
                raise Exception('The division index ({}) must be a valid index for the division list {}.'.format(div_idx, samples_division_list))

        # Transform class name to index
        subject_set = utils.extract_fold_list(fold_list, self.subject_list, self.nb_subject_per_fold)
        class_to_idx = {}
        for class_idx, subject in enumerate(subject_set):
            class_to_idx[subject] = class_idx

        samples = []
        stillclass_to_sampleidx = {}
        tbar = tqdm.tqdm(subject_set)
        for i, subject in enumerate(tbar):
            subject_video_path = os.path.join(self.video_source, subject)
            video_image_paths = utils.get_image_paths(subject_video_path)

            # Divide video samples if requested
            if samples_division_list:
                lower_div_idx = int(len(video_image_paths) * sum(samples_division_list[:div_idx]))
                upper_div_idx = int(len(video_image_paths) * sum(samples_division_list[:div_idx + 1]))
                video_image_paths = video_image_paths[lower_div_idx:upper_div_idx]

            still_image_path = os.path.join(self.still_source, subject + '_0000.JPG')
            if not os.path.isfile(still_image_path):
                raise Exception('Still image not found at {}'.format(still_image_path))

            if video_only:
                paths = video_image_paths
            else:
                paths = [still_image_path] + video_image_paths
                # (class_idx, sample_idx)
                stillclass_to_sampleidx[class_to_idx[subject]] = len(samples)

            for path in paths:
                item = (path, class_to_idx[subject])
                samples.append(item)

        self.stillclass_to_sampleidx = stillclass_to_sampleidx

        self.classes = subject_set
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.preloaded = False
        if preload:
            self.images = {}
            for path, lbl in self.samples:
                img = Image.open(path)
                self.images[path] = img.copy()
            self.preloaded = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        """
                Args:
                    index (int): Index

                Returns:
                    tuple: (sample, target) where target is class_index of the target class.
                """

        path, label = self.samples[idx]
        if self.preloaded:
            sample = self.images[path]
        else:
            sample = default_loader(path)

        if self.transform:
            sample = self.transform(sample)

        return sample, label
