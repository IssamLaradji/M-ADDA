
import torch
from torch.utils import data
from torchvision import datasets

import utils
import dataset_utils
from dataset_utils import sampler
from dataset_utils import coxs2v, lfw, vggface2

def Get_ImageFolderLoader(data_dir,
                 data_transform,
                 people_per_batch,
                 images_per_person):

    train_set = datasets.ImageFolder(data_dir, transform=data_transform)

    batch_sampler = sampler.Random_BalancedBatchSampler(train_set,
                                                        people_per_batch,
                                                        images_per_person, max_batches=1000)

    return torch.utils.data.DataLoader(train_set,
                                       num_workers=8,
                                       batch_sampler=batch_sampler,
                                       pin_memory=True)


def Get_PairsImageFolderLoader(data_dir,
                               pairs_file,
                               data_transform,
                               batch_size,
                               preload=False):

    num_workers = 2 if preload else 4

    test_set = dataset_utils.dataset.PairsDataset(data_dir,
                                     pairs_file,
                                     transform=data_transform,
                                     preload=preload)
    return torch.utils.data.DataLoader(test_set,
                                       num_workers=num_workers,
                                       batch_size=batch_size)


def Get_TestDataloaders(config,
                        data_transform,
                        batch_size,
                        folds,
                        nrof_folds,
                        is_vggface2=False,
                        is_lfw=False,
                        is_cox_video1=False,
                        is_cox_video2=False,
                        is_cox_video3=False,
                        is_cox_video4=False):

    test_loaders_list = []

    # VGGFACE2 dataset
    if is_vggface2:
        data_loader = vggface2.get_vggface2_testset(config.dataset.vggface2.test_dir,
                                           config.dataset.vggface2.pairs_file,
                                           data_transform,
                                           batch_size,
                                           preload=False)
        test_loaders_list.append(('vggface2', data_loader))

    # LFW dataset
    if is_lfw:
        data_loader = lfw.get_lfw_testset(config.dataset.lfw.test_dir,
                                      config.dataset.lfw.pairs_file,
                                      data_transform,
                                      batch_size,
                                      preload=False)
        test_loaders_list.append(('lfw', data_loader))

    # COXS2V dataset
    if is_cox_video1:
        data_loader = coxs2v.get_coxs2v_testset(config.dataset.coxs2v.still_dir,
                                         config.dataset.coxs2v.video1_dir,
                                         config.dataset.coxs2v.video1_pairs,
                                         folds,
                                         nrof_folds,
                                         data_transform,
                                         batch_size)
        test_loaders_list.append(('cox_video1', data_loader))

    if is_cox_video2:
        data_loader = coxs2v.get_coxs2v_testset(config.dataset.coxs2v.still_dir,
                                         config.dataset.coxs2v.video2_dir,
                                         config.dataset.coxs2v.video2_pairs,
                                         folds,
                                         nrof_folds,
                                         data_transform,
                                         batch_size)
        test_loaders_list.append(('cox_video2', data_loader))

    if is_cox_video3:
        data_loader = coxs2v.get_coxs2v_testset(config.dataset.coxs2v.still_dir,
                                         config.dataset.coxs2v.video3_dir,
                                         config.dataset.coxs2v.video3_pairs,
                                         folds,
                                         nrof_folds,
                                         data_transform,
                                         batch_size)
        test_loaders_list.append(('cox_video3', data_loader))

    if is_cox_video4:
        data_loader = coxs2v.get_coxs2v_testset(config.dataset.coxs2v.still_dir,
                                         config.dataset.coxs2v.video4_dir,
                                         config.dataset.coxs2v.video4_pairs,
                                         folds,
                                         nrof_folds,
                                         data_transform,
                                         batch_size)
        test_loaders_list.append(('cox_video4', data_loader))

    return test_loaders_list


def Get_TrainDataloaders(config,
                         data_transform,
                         people_per_batch,
                         images_per_person,
                         folds,
                         nrof_folds,
                         is_vggface2=False,
                         is_cox_video1=False,
                         is_cox_video2=False,
                         is_cox_video3=False,
                         is_cox_video4=False):

    train_loaders_list = []

    # VGGFACE2 dataset
    if is_vggface2:
        data_loader = vggface2.get_vggface2_trainset(config.dataset.vggface2.train_dir,
                                                     data_transform,
                                                     people_per_batch,
                                                     images_per_person)
        train_loaders_list.append(('vggface2', data_loader))

    # COXS2V dataset
    if is_cox_video1:
        data_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                 config.dataset.coxs2v.video1_dir,
                                                 config.dataset.coxs2v.video1_pairs,
                                                 folds,
                                                 nrof_folds,
                                                 data_transform,
                                                 people_per_batch,
                                                 images_per_person)
        train_loaders_list.append(('cox_video1', data_loader))

    if is_cox_video2:
        data_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                 config.dataset.coxs2v.video2_dir,
                                                 config.dataset.coxs2v.video2_pairs,
                                                 folds,
                                                 nrof_folds,
                                                 data_transform,
                                                 people_per_batch,
                                                 images_per_person)
        train_loaders_list.append(('cox_video2', data_loader))

    if is_cox_video3:
        data_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                 config.dataset.coxs2v.video3_dir,
                                                 config.dataset.coxs2v.video3_pairs,
                                                 folds,
                                                 nrof_folds,
                                                 data_transform,
                                                 people_per_batch,
                                                 images_per_person)
        train_loaders_list.append(('cox_video3', data_loader))

    if is_cox_video4:
        data_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                 config.dataset.coxs2v.video4_dir,
                                                 config.dataset.coxs2v.video4_pairs,
                                                 folds,
                                                 nrof_folds,
                                                 data_transform,
                                                 people_per_batch,
                                                 images_per_person)
        train_loaders_list.append(('cox_video4', data_loader))

    return train_loaders_list