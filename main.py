import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from models import Generator, Discriminator, VGG16
from model_wrapper import ModelWrapper
import data

if __name__ == '__main__':
    # Utilize gpus 3
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2, 3'

    # Init models, optimizers and dataset
    generator = nn.DataParallel(Generator())
    print('Number of generator parameters', sum(p.numel() for p in generator.parameters()))
    discriminator = nn.DataParallel(Discriminator())
    print('Number of discriminator parameters', sum(p.numel() for p in discriminator.parameters()))

    # Init optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    # Init dataset
    training_dataset = DataLoader(
        data.Places365(path_to_index_file='/home/creich/places365_standard', index_file_name='train.txt'),
        batch_size=60, num_workers=60, shuffle=True,
        collate_fn=data.image_label_list_of_masks_collate_function)
    validation_dataset_fid = DataLoader(
        data.Places365(path_to_index_file='/home/creich/places365_standard', index_file_name='val.txt',
                       max_length=6000, validation=True),
        batch_size=60, num_workers=60, shuffle=False,
        collate_fn=data.image_label_list_of_masks_collate_function)
    validation_dataset = data.Places365(path_to_index_file='/home/creich/places365_standard', index_file_name='val.txt',
                                        test=True)
    # Init model wrapper
    model_wrapper = ModelWrapper(generator=generator,
                                 discriminator=discriminator,
                                 vgg16=nn.DataParallel(VGG16()),
                                 training_dataset=training_dataset,
                                 validation_dataset=validation_dataset,
                                 validation_dataset_fid=validation_dataset_fid,
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer)
    # Perform training
    model_wrapper.inference()
    model_wrapper.train(epochs=100, device='cuda')
