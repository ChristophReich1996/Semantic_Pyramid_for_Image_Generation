from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms.functional as TVF

import misc


class PseudoDataset(Dataset):
    '''
    Class implements a pseudo dataset to test the training loop.
    '''

    def __init__(self, length: int = 10000, image_size: Tuple[int, int, int] = (1, 256, 256)) -> None:
        '''
        Constructor
        :param length: (int) Length of the dataset
        :param image_size: (Tuple[int, int, int]) Image size to be generated
        '''
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


class TinyImageNet(Dataset):
    '''
    Implementation of the tiny image net dataset
    '''

    def __init__(self, path: str = '', resolution: Tuple[int, int] = (256, 256), image_format: str = 'jpeg') -> None:
        '''
        Constructor
        :param path: (str) Path to images
        :param resolution: (Tuple[int, int]) Desired resolution
        :param image_format: (str) Image file format to detect in path
        '''
        # Save parameter
        self.resolution = resolution
        self.path = path
        # Detect all images in path and subdirectories
        image_format = image_format.lower()
        self.files = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if image_format in file.lower():
                    self.files.append(os.path.join(root, file))

    def __len__(self) -> int:
        '''
        Method returns the length of the dataset
        :return: (int) Length of the dataset
        '''
        return len(self.files)

    def __getitem__(self, item) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        '''
        Method returns one instance of the dataset and a list of corresponding random masks.
        :param item: (int) Index of the dataset
        :return: (torch.Tensor) Loaded image
        '''
        # Load image
        image = Image.open(os.path.join(self.path, self.files[item]))
        # Image to tensor
        image = TVF.to_tensor(image)
        # Reshape image
        image = F.interpolate(image.unsqueeze(dim=0), size=self.resolution, mode='bilinear',
                              align_corners=False).squeeze(dim=0)
        # Add rgb channels if needed
        if image.shape[0] == 1:
            image = image.repeat_interleave(repeats=3, dim=0)
        # Return image and random masks
        return image, misc.get_masks_for_training()


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
