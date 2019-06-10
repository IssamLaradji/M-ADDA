
import torch
from dataset_utils import dataset
from dataset_utils import sampler


def get_coxs2v_trainset(still_dir,
                        video_dir,
                        pairs_file,
                        folds,
                        nrof_folds,
                        data_transform,
                        people_per_batch,
                        images_per_person,
                        video_only=False,
                        samples_division_list=None,  # [0.6, 0.4]
                        div_idx: int = -1):

    # Set up train loader
    print('TRAIN SET COXS2V:\t{}'.format(video_dir))
    train_set = dataset.DatasetS2V(still_dir,
                                   video_dir,
                                   pairs_file,
                                   data_transform,
                                   folds,
                                   video_only=video_only,
                                   num_folds=nrof_folds,
                                   samples_division_list=samples_division_list,  # [0.4, 0.6]
                                   div_idx=div_idx)

    batch_sampler = sampler.Random_S2VBalancedBatchSampler(train_set,
                                                   people_per_batch,
                                                   images_per_person)

    return torch.utils.data.DataLoader(train_set,
                                       num_workers=8,
                                       batch_sampler=batch_sampler,
                                       pin_memory=False)

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
    test_set = dataset.PairsDatasetS2V(still_dir,
                                       video_dir,
                                       pairs_file,
                                       data_transform,
                                       folds,
                                       num_folds=nrof_folds,
                                       preload=preload)
    return torch.utils.data.DataLoader(test_set, num_workers=num_workers, batch_size=batch_size)