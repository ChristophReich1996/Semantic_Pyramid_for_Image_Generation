import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from scipy.linalg import sqrtm

import misc


class InceptionNetworkFID(nn.Module):
    '''
    This class implements a pre trained inception network for getting the output of layer 7c
    https://github.com/hukkelas/pytorch-frechet-inception-distance/blob/master/fid.py
    '''

    def __init__(self):
        # Call super constructor
        super(InceptionNetworkFID, self).__init__()
        # Init pre trained inception net
        self.inception_net = torchvision.models.inception_v3(pretrained=True, transform_input=True)
        # Init hook to get intermediate output
        self.inception_net.Mixed_7c.register_forward_hook(self.output_hook)

    def output_hook(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        # Save output
        self.output_7c = output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        Forward method
        :param input: (torch.Tensor) Input tensor normalized to a range of [-1, 1]
        :return: (torch.Tensor) Rescaled intermediate output of layer 7c of the inception net
        '''
        # Forward pass of inception to get produce output
        self.inception_net(input)
        # Get intermediate output and downscale tensor to get a tensor of shape (batch size, 2024, 1, 1)
        output = F.adaptive_avg_pool2d(self.output_7c, (1, 1))
        # Reshape output to shape (batch size, 2024)
        output = output.view(input.shape[0], 2048)
        return output


@torch.no_grad()
def frechet_inception_distance(dataset_real: DataLoader, generator: nn.Module, vgg16: nn.Module,
                               device: str = 'cuda') -> float:
    '''
    Function computes the frechet inception distance
    :param dataset_real: (Dataset) Dataset including real samples
    :param dataset_fake: (Datasets.GeneratorDataset) Dataset samples from generator network
    :param batch_size: (int) Batch size to be utilized
    :param device: (str) Device to use
    :return: (float) FID score
    '''
    # Init inception net
    inception_net = InceptionNetworkFID().to(device)
    # Get real activations
    real_activations = []
    fake_activations = []
    for images, labels, masks in dataset_real:
        # Data to device
        images = images.to(device)
        del labels
        for index in range(len(masks)):
            masks[index] = masks[index].to(device)
        # Normalize images
        images_normalized = misc.normalize_0_1_batch(images)
        # Reshape
        if images_normalized.shape[2] != 299 or images_normalized.shape[3] != 299:
            images_normalized = nn.functional.interpolate(images_normalized, size=(299, 299), mode='bilinear',
                                                          align_corners=False)
        # Get activations
        real_activations.append(inception_net(images_normalized).detach().cpu())
        # Get fake images
        # Get features of images from vgg16 model
        with torch.no_grad():
            features_real = vgg16(images)
        # Generate random noise vector
        noise_vector = torch.randn((images.shape[0],
                                    generator.module.latent_dimensions
                                    if isinstance(generator, nn.DataParallel) else generator.latent_dimensions),
                                   dtype=torch.float32, device=device, requires_grad=True)
        # Generate fake images
        images_fake = generator(input=noise_vector, features=features_real, masks=masks)
        # Normalize images
        images_fake_normalized = misc.normalize_0_1_batch(images_fake)
        # Reshape
        if images_fake_normalized.shape[2] != 299 or images_fake_normalized.shape[3] != 299:
            images_fake_normalized = nn.functional.interpolate(images_fake_normalized, size=(299, 299), mode='bilinear',
                                                               align_corners=False)
        # Get activation
        fake_activations.append(inception_net(images_fake_normalized).detach().cpu())
    # Make one big numpy array by concat at batch dim
    real_activations = torch.cat(real_activations, dim=0).numpy()
    # Make again one big numpy array
    fake_activations = torch.cat(fake_activations, dim=0).numpy()
    # Calc statistics of real activations
    real_mu = np.mean(real_activations, axis=0)
    real_cov = np.cov(real_activations, rowvar=False)
    # Calc statistics of fake activations
    fake_mu = np.mean(fake_activations, axis=0)
    fake_cov = np.cov(fake_activations, rowvar=False)
    # Check that mu and cov arrays of real and fake have the same shapes
    assert real_mu.shape == fake_mu.shape
    assert real_cov.shape == fake_cov.shape
    # Calc diff of mu real and fake
    diff = real_mu - fake_mu
    # Square diff
    diff_squared = diff @ diff
    # Calc cov mean of fake and real cov
    cov_mean, _ = sqrtm(real_cov @ fake_cov, disp=False)
    # Remove imag path of cov mean
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    # Calc FID
    fid = diff_squared + np.trace(real_cov) + np.trace(fake_cov) - 2 * np.trace(cov_mean)
    return fid
