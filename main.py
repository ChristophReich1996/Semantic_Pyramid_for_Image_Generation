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
    generator = nn.DataParallel(Generator(channels_factor=2))
    print('Number of generator parameters', sum(p.numel() for p in generator.parameters()))
    discriminator = nn.DataParallel(Discriminator())
    print('Number of discriminator parameters', sum(p.numel() for p in discriminator.parameters()))

    # Init optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    # Init dataset
    training_dataset = DataLoader(
        data.Places365(path_to_index_file='/home/creich/places365_standard', index_file_name='train.txt'),
        batch_size=90, num_workers=90, shuffle=True,
        collate_fn=data.image_label_list_of_masks_collate_function)
    validation_dataset = DataLoader(
        data.Places365(path_to_index_file='/home/creich/places365_standard', index_file_name='train.txt'),
        batch_size=90, num_workers=90, shuffle=True,
        collate_fn=data.image_label_list_of_masks_collate_function)
    # Init model wrapper
    model_wrapper = ModelWrapper(generator=generator,
                                 discriminator=discriminator,
                                 vgg16=nn.DataParallel(VGG16()),
                                 training_dataset=training_dataset,
                                 validation_dataset=validation_dataset,
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer)
    # Perform training
    model_wrapper.train(epochs=2, device='cuda')
