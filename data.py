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


def tensor_list_of_masks_collate_function(batch: List[Tuple[torch.Tensor, List[torch.Tensor]]]) -> Tuple[
    torch.Tensor, List[torch.Tensor]]:
    '''
    Function batches a list of given samples. Each samples contains an image and a list of masks
    :param batch: (List[Tuple[torch.Tensor, List[torch.Tensor]]]) List of samples
    :return: (Tuple[torch.Tensor, List[torch.Tensor]]) Batched samples
    '''
    # Batching images
    images = torch.stack([instance[0] for instance in batch], dim=0)
    # Set requires grad
    images.requires_grad = True
    # Batching masks by iterating over all masks
    masks = []
    for mask_index in range(len(batch[0][1])):
        # Make batch
        mask = torch.stack([batch[batch_index][1][mask_index] for batch_index in range(len(batch))], dim=0)
        # Set requires grad
        mask.requires_grad = True
        # Save batched mask in list
        masks.append(mask)
    return images, masks
