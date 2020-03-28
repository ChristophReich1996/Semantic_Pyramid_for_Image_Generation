# TODO: Implement class-embeddings and truncation trick
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from models import Generator, Discriminator
from model_wrapper import ModelWrapper
import data

if __name__ == '__main__':
    # Utilize gpus 3
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 3'

    # Init models, optimizers and dataset
    generator = nn.DataParallel(Generator(channels_factor=2))
    print('Number of generator parameters', sum(p.numel() for p in generator.parameters()))
    discriminator = nn.DataParallel(Discriminator())
    print('Number of discriminator parameters', sum(p.numel() for p in discriminator.parameters()))

    # Init optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.003)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    # Init dataset
    training_dataset = DataLoader(data.TinyImageNet(path='/home/creich/tiny-image-net/tiny-imagenet-200/train'),
                                  batch_size=60, num_workers=40, shuffle=True,
                                  collate_fn=data.tensor_list_of_masks_collate_function)
    validation_dataset = DataLoader(data.TinyImageNet(path='/home/creich/tiny-image-net/tiny-imagenet-200/val'),
                                    batch_size=60, num_workers=40, shuffle=True,
                                    collate_fn=data.tensor_list_of_masks_collate_function)
    test_dataset = DataLoader(data.TinyImageNet(path='/home/creich/tiny-image-net/tiny-imagenet-200/test'),
                              batch_size=60, num_workers=40, shuffle=True,
                              collate_fn=data.tensor_list_of_masks_collate_function)
    # Init model wrapper
    model_wrapper = ModelWrapper(generator=generator, discriminator=discriminator, training_dataset=training_dataset,
                                 test_dataset=test_dataset, validation_dataset=validation_dataset,
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer)
    # Perform training
    model_wrapper.train(training_iterations=100000, device='cuda')
