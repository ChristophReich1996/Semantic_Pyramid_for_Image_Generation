from argparse import ArgumentParser

# Process command line arguments
parser = ArgumentParser()

parser.add_argument('--train', default=False, action='store_true',
                    help='Train network')

parser.add_argument('--test', default=False, action='store_true',
                    help='Test network')

parser.add_argument('--batch_size', type=int, default=20,
                    help='Batch size of the training and test set (default=60)')

parser.add_argument('--lr', type=float, default=1e-05,
                    help='Main learning rate of the adam optimizer (default=1e-04)')

parser.add_argument('--channel_factor', type=float, default=1.0,
                    help='Channel factor adopts the number of channels utilized in the U-Net (default=1)')

parser.add_argument('--device', type=str, default='cuda',
                    help='Device to use (default=cuda)')

parser.add_argument('--gpus_to_use', type=str, default='0',
                    help='Indexes of the GPUs to be use (default=0)')

parser.add_argument('--use_data_parallel', default=False, action='store_true',
                    help='Use multiple GPUs (default=0 (False))')

parser.add_argument('--load_generator_network', type=str, default=None,
                    help='Name of the generator network the be loaded from model file (.pt) (default=None)')

parser.add_argument('--load_discriminator_network', type=str, default=None,
                    help='Name of the discriminator network the be loaded from model file (.pt) (default=None)')

parser.add_argument('--load_pretrained_vgg16', type=str, default='pre_trained_models/vgg_places_365_fine_tuned.pt',
                    help='Name of the pretrained (places365) vgg16 network the be loaded from model file (.pt)')

parser.add_argument('--path_to_places365', type=str, default='places365_standard',
                    help='Path to places365 dataset.')

parser.add_argument('--epochs', type=int, default=50,
                    help='Epochs to perform while training (default=100)')

args = parser.parse_args()

import os

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_to_use

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import Generator, Discriminator, VGG16
from model_wrapper import ModelWrapper
import data

if __name__ == '__main__':
    # Init models
    if args.load_generator_network is None:
        generator = Generator(channels_factor=args.channel_factor)
    else:
        generator = torch.load(args.load_generator_network)
        if isinstance(generator, nn.DataParallel):
            generator = generator.module
    if args.load_discriminator_network is None:
        discriminator = Discriminator(channel_factor=args.channel_factor)
    else:
        discriminator = torch.load(args.load_discriminator_network)
        if isinstance(discriminator, nn.DataParallel):
            discriminator = discriminator.module
    vgg16 = VGG16()
    vgg16.load_state_dict(torch.load(args.load_pretrained_vgg16, map_location="cpu"))
    # Init data parallel
    if args.use_data_parallel:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        vgg16 = nn.DataParallel(vgg16)

    # Init optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.1 * args.lr)
    # Print number of network parameters
    print('Number of generator parameters', sum(p.numel() for p in generator.parameters()))
    print('Number of discriminator parameters', sum(p.numel() for p in discriminator.parameters()))

    # Init dataset
    training_dataset = DataLoader(
        data.Places365(path_to_index_file=args.path_to_places365, index_file_name='train.txt'),
        batch_size=args.batch_size, num_workers=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=data.image_label_list_of_masks_collate_function)
    validation_dataset_fid = DataLoader(
        data.Places365(path_to_index_file=args.path_to_places365, index_file_name='val.txt',
                       max_length=6000, validation=True),
        batch_size=2 * args.batch_size, num_workers=2 * args.batch_size, shuffle=True,
        collate_fn=data.image_label_list_of_masks_collate_function)
    validation_dataset = data.Places365(path_to_index_file=args.path_to_places365, index_file_name='val.txt')
    # Init model wrapper
    model_wrapper = ModelWrapper(generator=generator,
                                 discriminator=discriminator,
                                 vgg16=vgg16,
                                 training_dataset=training_dataset,
                                 validation_dataset=validation_dataset,
                                 validation_dataset_fid=validation_dataset_fid,
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer)
    # Perform training
    if args.train:
        model_wrapper.train(epochs=args.epochs, device=args.device)
    # Perform testing
    if args.test:
        print('FID=', model_wrapper.validate(device=args.device))
        model_wrapper.inference(device=args.device)
