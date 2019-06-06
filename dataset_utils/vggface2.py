
from dataset_utils import Get_ImageFolderLoader, Get_PairsImageFolderLoader


def get_vggface2_trainset(train_dir,
                 data_transform,
                 people_per_batch,
                 images_per_person):


    # Set up train loader
    print('TRAIN SET vggface2:\t{}'.format(train_dir))
    return Get_ImageFolderLoader(train_dir,
                                 data_transform,
                                 people_per_batch,
                                 images_per_person)

def get_vggface2_testset(test_dir,
                         pairs_file,
                         data_transform,
                         batch_size,
                         preload=False):


    print('TEST SET vggface2:\t{}'.format(test_dir))
    return Get_PairsImageFolderLoader(test_dir,
                                      pairs_file,
                                      data_transform,
                                      batch_size,
                                      preload=preload)