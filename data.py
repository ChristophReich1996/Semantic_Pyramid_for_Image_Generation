from typing import Tuple, List, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms.functional as TVF
import pandas as pd

import misc


class Places365(Dataset):

    def __init__(self, path_to_index_file: str = '', index_file_name: str = 'train.txt',
                 return_masks: bool = True) -> None:
        # Save parameter
        self.path_to_index_file = path_to_index_file
        self.return_masks = return_masks
        # Get index file
        self.file_paths = pd.read_csv(os.path.join(path_to_index_file, index_file_name)).values[:, 0]
        self.file_paths.sort()
        # Make dict of labels
        self.label_dict = dict()
        for file_path in self.file_paths:
            folder = file_path.split('/')[1]
            if folder not in self.label_dict:
                self.label_dict[folder] = len(self.label_dict)

    def __len__(self) -> int:
        return len(self.file_paths)

    def set_return_masks(self, flag: bool) -> None:
        self.return_masks = flag

    def __getitem__(self, item) -> Union[
        Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]:
        '''
        Method returns one instance of the dataset and a list of corresponding random masks.
        :param item: (int) Index of the dataset
        :return: (torch.Tensor) Loaded image
        '''
        image = Image.open(os.path.join(self.path_to_index_file, self.file_paths[item]))
        # Image to tensor
        image = TVF.to_tensor(image)
        # Normalize image to a range of 0 to 1
        image.sub_(image.min()).div_(image.max() - image.min())
        # Make label
        label = torch.zeros(len(self.label_dict), dtype=torch.long)
        label[self.label_dict[self.file_paths[item].split('/')[1]]] = 1
        # Get random masks
        if self.return_masks:
            masks = misc.get_masks_for_training()
            return image, label, masks
        else:
            return image, label


def image_label_list_of_masks_collate_function(batch: List[Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]]) -> \
        Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    '''
    Function batches a list of given samples. Each samples contains an image and a list of masks
    :param batch: (List[Tuple[torch.Tensor, List[torch.Tensor]]]) List of samples
    :return: (Tuple[torch.Tensor, List[torch.Tensor]]) Batched samples
    '''
    # Batching images
    images = torch.stack([instance[0] for instance in batch], dim=0)
    # Batching labels
    labels = torch.stack([instance[1] for instance in batch], dim=0)
    # Set requires grad
    images.requires_grad = True
    # Batching masks by iterating over all masks
    masks = []
    for mask_index in range(len(batch[0][2])):
        # Make batch
        mask = torch.stack([batch[batch_index][2][mask_index] for batch_index in range(len(batch))], dim=0)
        # Set requires grad
        mask.requires_grad = True
        # Save batched mask in list
        masks.append(mask)
    return images, labels, masks
