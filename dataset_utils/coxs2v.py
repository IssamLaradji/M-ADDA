
import torch
from dataset_utils import Get_ImageFolderLoader, Get_PairsImageFolderLoader
from dataset_utils import PairsDataset, PairsDatasetS2V, DatasetS2V
from dataset_utils import Random_BalancedBatchSampler, Random_S2VBalancedBatchSampler

def get_coxs2v_trainset(still_dir,
                          video_dir,
                          pairs_file,
                          folds,
                          nrof_folds,
                          data_transform,
                          people_per_batch,
                          images_per_person):


    # Set up train loader
    print('TRAIN SET COXS2V:\t{}'.format(video_dir))
    train_set = DatasetS2V(still_dir,
                           video_dir,
                           pairs_file,
                           data_transform,
                           folds,
                           num_folds=nrof_folds)

    batch_sampler = Random_S2VBalancedBatchSampler(train_set,
                                                   people_per_batch,
                                                   images_per_person,
                                                   max_batches=1000)

    return torch.utils.data.DataLoader(train_set,
                                       num_workers=8,
                                       batch_sampler=batch_sampler,
                                       pin_memory=True)

def get_coxs2v_testset(still_dir,
                       video_dir,
                       pairs_file,
                       folds,
                       nrof_folds,
                       data_transform,
                       batch_size,
                       preload=False):

    num_workers = 2 if preload else 4


    print('TEST SET COXS2V:\t{}'.format(video_dir))
    test_set = PairsDatasetS2V(still_dir,
                               video_dir,
                               pairs_file,
                               data_transform,
                               folds,
                               num_folds=nrof_folds,
                               preload=preload)
    return torch.utils.data.DataLoader(test_set, num_workers=num_workers, batch_size=batch_size)