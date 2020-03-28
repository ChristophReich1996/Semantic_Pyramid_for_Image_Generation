from typing import Tuple, List

import torch
from torch.utils.data import Dataset

import misc


class PseudoDataset(Dataset):
    '''
    Class implements a pseudo dataset to test the training loop.
    '''

    def __init__(self, length: int = 10000, image_size: Tuple[int, int, int] = (1, 256, 256)):
        '''
        Constructor
        :param length: (int) Length of the dataset
        :param image_size: (Tuple[int, int, int]) Image size to be generated
        '''
        # Call super constructor
        super(PseudoDataset, self).__init__()
        # Save parameters
        self.length = length
        self.image_size = image_size

    def __len__(self) -> int:
        '''
        Method returns the length of the dataset
        :return: (int) Length of the dataset
        '''
        return self.length

    def __getitem__(self, item) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        '''
        Method returns randomly generated images including a Gaussian noise and a list of corresponding random masks.
        :param item: (int) Index to get
        :return: (Tuple[torch.Tensor, List[torch.Tensor]]) Image and list of masks
        '''
        # Raise error if index out of bounce
        if item > len(self) or item < 0:
            raise IndexError()
        else:
            # Generate and return pseudo data
            return torch.randn(self.image_size), misc.get_masks_for_training()
