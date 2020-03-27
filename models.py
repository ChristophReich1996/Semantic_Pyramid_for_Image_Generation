from typing import List, Union

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import torchvision
import math


class Generator(nn.Module):
    '''
    Generator network
    '''

    def __init__(self, output_channels: int = 1, latent_dimensions: int = 100,
                 channels_factor: Union[int, float] = 1) -> None:
        '''
        Constructor method
        :param output_channels: (int) Number of output channels (1 = grayscale, 3 = rgb)
        :param latent_dimensions: (int) Latent dimension size
        :param channels_factor: (int, float) Channel factor to adopt the feature size in each layer
        '''
        super(Generator, self).__init__()
        # Init linear input layers
        self.input_path = nn.ModuleList([
            LinearBlock(in_features=latent_dimensions, out_features=int(128 // channels_factor), feature_size=1000),
            LinearBlock(in_features=int(128 // channels_factor), out_features=int(128 // channels_factor),
                        feature_size=4096),
            nn.Linear(in_features=int(128 // channels_factor), out_features=512 * 4 * 4),
            nn.LeakyReLU(negative_slope=0.2)
        ])
        # Init main residual path
        self.main_path = nn.ModuleList([
            ResidualBlock(in_channels=int(512 // channels_factor), out_channels=int(512 // channels_factor),
                          feature_channels=512),
            ResidualBlock(in_channels=int(512 // channels_factor), out_channels=int(512 // channels_factor),
                          feature_channels=512),
            ResidualBlock(in_channels=int(512 // channels_factor), out_channels=int(256 // channels_factor),
                          feature_channels=256),
            SelfAttention(channels=int(256 // channels_factor)),
            ResidualBlock(in_channels=int(256 // channels_factor), out_channels=int(128 // channels_factor),
                          feature_channels=128),
            ResidualBlock(in_channels=int(128 // channels_factor), out_channels=int(64 // channels_factor),
                          feature_channels=64)
        ])
        # Init final block
        self.final_block = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(int(64 // channels_factor)),
            spectral_norm(nn.Conv2d(in_channels=int(64 // channels_factor), out_channels=int(64 // channels_factor),
                                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(
                nn.Conv2d(in_channels=int(64 // channels_factor), out_channels=output_channels, kernel_size=(1, 1),
                          stride=(1, 1), padding=(0, 0), bias=True))
        )

    def forward(self, input: torch.Tensor, masked_features: List[torch.Tensor]) -> torch.Tensor:
        '''
        Forward pass
        :param input: (torch.Tensor) Input latent tensor
        :param masked_features: (List[torch.Tensor]) List of vgg16 features
        :return: (torch.Tensor) Generated output image
        '''
        # Input path
        for index, layer in enumerate(self.input_path):
            if index == 0:
                output = layer(input, masked_features.pop(-1))
            elif index == 1:
                output = layer(output, masked_features.pop(-1))
            else:
                output = layer(output)
        # Reshaping
        output = output.view(output.shape[0], int(output.shape[1] // (4 ** 2)), 4, 4)
        # Main path
        for layer in self.main_path:
            if isinstance(layer, SelfAttention):
                output = layer(output)
            else:
                output = layer(output, masked_features.pop(-1))
        # Final block
        output = self.final_block(output)
        return output


class Discriminator(nn.Module):
    '''
    Discriminator network
    '''
    pass


class VGG16(nn.Module):
    '''
    Implementation of a pre-trained VGG 16 model which outputs intermediate feature activations of the model.
    '''

    def __init__(self, pretrained: bool = True) -> None:
        '''
        Constructor
        :param pretrained: (bool) True if the default pre trained vgg16 model pre trained in image net should be used
        '''
        # Call super constructor
        super(VGG16, self).__init__()
        # Load model from torchvision
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        # Init forward hooks of feature module
        for layer in self.vgg16.features:
            if isinstance(layer, nn.MaxPool2d):
                layer.register_forward_hook(self.store_activation)
        # Init forward nooks for fully connected layers
        self.vgg16.classifier[3].register_forward_hook(self.store_activation)
        self.vgg16.classifier[6].register_forward_hook(self.store_activation)

    def store_activation(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        '''
        Hook method to save output activation
        :param module: (nn.Module) Unused parameter
        :param input: (torch.Tensor) Unused parameter
        :param output: (torch.Tensor) Output tensor of the called module
        '''
        # Save activation in list
        self.feature_activations.append(output)

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        '''
        Forward pass of the model
        :param input: (torch.Tenor) Input tensor of shape (batch size, channels, height, width)
        :return: (List[torch.Tensor]) List of intermediate features in ascending oder w.r.t. the number VGG layer
        '''
        # Empty feature activation list
        self.feature_activations = []
        # Call forward pass of vgg 16 to trigger hooks
        self.vgg16(input)
        # Return activations
        return self.feature_activations


class SelfAttention(nn.Module):
    '''
    Self attention module proposed in: https://arxiv.org/pdf/1805.08318.pdf.
    '''

    def __init__(self, channels: int):
        '''
        Constructor
        :param channels: (int) Number of channels to be utilized
        '''
        # Call super constructor
        super(SelfAttention, self).__init__()
        # Init convolutions
        self.query_convolution = spectral_norm(
            nn.Conv2d(in_channels=channels, out_channels=channels // 8, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0), bias=True))
        self.key_convolution = spectral_norm(
            nn.Conv2d(in_channels=channels, out_channels=channels // 8, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0), bias=True))
        self.value_convolution = spectral_norm(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0), bias=True))
        # Init gamma parameter
        self.gamma = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass
        :param input: (torch.Tensor) Input tensor
        :return: (torch.Tensor) Output tensor
        '''
        # Save input shape
        batch_size, channels, height, width = input.shape
        # Mappings
        query_mapping = self.query_convolution(input)
        key_mapping = self.key_convolution(input)
        value_mapping = self.value_convolution(input)
        # Reshape and transpose query mapping
        query_mapping = query_mapping.view(batch_size, -1, height * width).permute(0, 2, 1)
        # Reshape key mapping
        key_mapping = key_mapping.view(batch_size, -1, height * width)
        # Calc attention maps
        attention = F.softmax(torch.bmm(query_mapping, key_mapping), dim=1)
        # Reshape value mapping
        value_mapping = value_mapping.view(batch_size, -1, height * width)
        # Attention features
        attention_features = torch.bmm(value_mapping, attention)
        # Reshape to original shape
        attention_features = attention_features.view(batch_size, channels, height, width)
        # Residual mapping and gamma multiplication
        output = self.gamma * attention_features + input
        return output


class ResidualBlock(nn.Module):
    '''
    Residual block
    '''

    def __init__(self, in_channels: int, out_channels: int, feature_channels: int) -> None:
        # Call super constructor
        super(ResidualBlock, self).__init__()
        # Init main operations
        self.main_block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
        )
        # Init residual mapping
        self.residual_mapping = spectral_norm(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0), bias=True))
        # Init upsampling
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        # Init convolution for mapping the masked features
        self.masked_feature_mapping = spectral_norm(nn.Conv2d(in_channels=feature_channels, out_channels=out_channels,
                                                              kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                                              bias=True))

    def forward(self, input: torch.Tensor, masked_features: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass
        :param input: (torch.Tensor) Input tensor
        :param masked_features: (torch.Tensor) Masked feature tensor form vvg16
        :return: (torch.Tensor) Output tensor
        '''
        # Main path
        output_main = self.main_block(input)
        # Residual mapping
        output_residual = self.residual_mapping(input)
        output_main = output_main + output_residual
        # Upsampling
        output_main = self.upsampling(output_main)
        # Feature path
        mapped_features = self.masked_feature_mapping(masked_features)
        # Addition step
        output = output_main + mapped_features
        return output


class LinearBlock(nn.Module):

    def __init__(self, in_features: int, out_features: int, feature_size: int) -> None:
        '''
        Constructor
        :param in_features:
        :param out_features:
        '''
        # Call super constructor
        super(LinearBlock, self).__init__()
        # Init linear layer and activation
        self.main_block = nn.Sequential(
            spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=True)),
            nn.LeakyReLU(negative_slope=0.2)
        )
        # Init mapping the masked features
        self.masked_feature_mapping = nn.Linear(in_features=feature_size, out_features=out_features, bias=True)

    def forward(self, input: torch.Tensor, masked_features: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass
        :param input: (torch.Tensor) Input tensor
        :param masked_features: (torch.Tensor) Masked feature tensor form vvg16
        :return: (torch.Tensor) Output tensor
        '''
        # Main path
        output_main = self.main_block(input)
        # Feature path
        mapped_features = self.masked_feature_mapping(masked_features)
        # Addition step
        output = output_main + mapped_features
        return output


if __name__ == '__main__':
    vgg16 = VGG16()
    generator = Generator()
    output = generator(torch.rand(1, 100), vgg16(torch.randn(1, 3, 256, 256)))
    print(output.shape)
