
from dataset_utils import Get_PairsImageFolderLoader

def get_lfw_testset(test_dir,
                 pairs_file,
                 data_transform,
                 batch_size,
                 preload=False):

    # Set up train loader
    print('TEST SET lfw:\t{}'.format(test_dir))
    return Get_PairsImageFolderLoader(test_dir,
                                      pairs_file,
                                      data_transform,
                                      batch_size,
                                      preload=preload)

