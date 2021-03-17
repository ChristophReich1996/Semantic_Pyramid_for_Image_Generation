from typing import List, Union

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import torchvision


class Generator(nn.Module):
    '''
    Generator network
    '''

    def __init__(self, out_channels: int = 3, latent_dimensions: int = 128,
                 channels_factor: Union[int, float] = 1, number_of_classes: int = 365) -> None:
        '''
        Constructor method
        :param out_channels: (int) Number of output channels (1 = grayscale, 3 = rgb)
        :param latent_dimensions: (int) Latent dimension size
        :param channels_factor: (int, float) Channel factor to adopt the channel size in each layer
        :param number_of_classes: (int) Number of classes in class tensor
        '''
        super(Generator, self).__init__()
        # Save parameters
        self.latent_dimensions = latent_dimensions
        # Init linear input layers
        self.input_path = nn.ModuleList([
            LinearBlock(in_features=latent_dimensions, out_features=256, feature_size=365),
            LinearBlock(in_features=256, out_features=256, feature_size=4096),
            nn.Sequential(spectral_norm(nn.Linear(in_features=256, out_features=int(512 // channels_factor) * 4 * 4)),
                          nn.LeakyReLU(negative_slope=0.2))
        ])
        # Init main residual path
        self.main_path = nn.ModuleList([
            GeneratorResidualBlock(in_channels=int(512 // channels_factor), out_channels=int(512 // channels_factor),
                                   feature_channels=513, number_of_classes=number_of_classes),
            GeneratorResidualBlock(in_channels=int(512 // channels_factor), out_channels=int(512 // channels_factor),
                                   feature_channels=513, number_of_classes=number_of_classes),
            GeneratorResidualBlock(in_channels=int(512 // channels_factor), out_channels=int(256 // channels_factor),
                                   feature_channels=257, number_of_classes=number_of_classes),
            SelfAttention(channels=int(256 // channels_factor)),
            GeneratorResidualBlock(in_channels=int(256 // channels_factor), out_channels=int(128 // channels_factor),
                                   feature_channels=129, number_of_classes=number_of_classes),
            GeneratorResidualBlock(in_channels=int(128 // channels_factor), out_channels=int(64 // channels_factor),
                                   feature_channels=65, number_of_classes=number_of_classes)
        ])
        # Init final block
        self.final_block = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(int(64 // channels_factor)),
            spectral_norm(nn.Conv2d(in_channels=int(64 // channels_factor), out_channels=int(64 // channels_factor),
                                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(
                nn.Conv2d(in_channels=int(64 // channels_factor), out_channels=out_channels, kernel_size=(1, 1),
                          stride=(1, 1), padding=(0, 0), bias=False))
        )

    def forward(self, input: torch.Tensor, features: List[torch.Tensor],
                masks: List[torch.Tensor] = None, class_id: torch.Tensor = None) -> torch.Tensor:
        '''
        Forward pass
        :param input: (torch.Tensor) Input latent tensor
        :param features: (List[torch.Tensor]) List of vgg16 features
        :return: (torch.Tensor) Generated output image
        '''
        # Init depth counter
        depth_counter = len(features) - 1
        # Input path
        for index, layer in enumerate(self.input_path):
            if index == 0:
                # Mask feature
                feature = features[depth_counter] * masks[depth_counter]
                output = layer(input, feature)
                depth_counter -= 1
            elif index == 1:
                # Mask feature
                feature = features[depth_counter] * masks[depth_counter]
                output = layer(output, feature)
                depth_counter -= 1
            else:
                output = layer(output)
        # Reshaping
        output = output.view(output.shape[0], int(output.shape[1] // (4 ** 2)), 4, 4)
        # Main path
        for layer in self.main_path:
            if isinstance(layer, SelfAttention):
                output = layer(output)
            else:
                # Mask feature and concat mask
                feature = features[depth_counter]
                mask = masks[depth_counter]
                feature = torch.cat((feature * mask, mask), dim=1)
                output = layer(output, feature, class_id)
                depth_counter -= 1
        # Final block
        output = self.final_block(output)
        return output.sigmoid()


class Discriminator(nn.Module):
    '''
    Discriminator network
    '''

    def __init__(self, in_channels: int = 3, channel_factor: Union[int, float] = 1, number_of_classes: int = 365):
        '''
        Constructor mehtod
        :param in_channels: (int) Number of input channels (grayscale = 1, rgb =3)
        :param channel_factor: (int, float) Channel factor to adopt the channel size in each layer
        '''
        # Call super constructor
        super(Discriminator, self).__init__()
        # Init layers
        self.layers = nn.Sequential(
            DiscriminatorResidualBlock(in_channels=in_channels, out_channels=int(64 // channel_factor)),
            DiscriminatorResidualBlock(in_channels=int(64 // channel_factor), out_channels=int(128 // channel_factor)),
            DiscriminatorResidualBlock(in_channels=int(128 // channel_factor), out_channels=int(256 // channel_factor)),
            SelfAttention(channels=int(256 // channel_factor)),
            DiscriminatorResidualBlock(in_channels=int(256 // channel_factor), out_channels=int(256 // channel_factor)),
            SelfAttention(channels=int(256 // channel_factor)),
            DiscriminatorResidualBlock(in_channels=int(256 // channel_factor), out_channels=int(256 // channel_factor)),
            DiscriminatorResidualBlock(in_channels=int(256 // channel_factor), out_channels=int(256 // channel_factor)),
            DiscriminatorResidualBlock(in_channels=int(256 // channel_factor), out_channels=int(256 // channel_factor)),
        )
        # Init classification layer
        self.classification = spectral_norm(
            nn.Linear(in_features=int(256 // channel_factor) * 2 * 2, out_features=1, bias=False))
        # Init embedding layer
        self.embedding = spectral_norm(nn.Embedding(num_embeddings=number_of_classes,
                                                    embedding_dim=int(256 // channel_factor) * 2 * 2))
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input: torch.Tensor, class_id: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass
        :param input: (torch.Tensor) Input image to be classified, real or fake. Image shape (batch size, 1 or 3, height, width)
        :return: (torch.Tensor) Output prediction of shape (batch size, 1)
        '''
        # Main path
        output = self.layers(input)
        # Reshape output into two dimensions
        output = output.flatten(start_dim=1)
        # Perform embedding
        output_embedding = self.embedding(class_id)
        output_embedding = (output.unsqueeze(dim=1) * output_embedding).sum(dim=1)
        # Classification path
        output = self.classification(output)
        return output + output_embedding


class VGG16(nn.Module):
    '''
    Implementation of a pre-trained VGG 16 model which outputs intermediate feature activations of the model.
    '''

    def __init__(self, path_to_pre_trained_model: str = None) -> None:
        '''
        Constructor
        :param pretrained: (bool) True if the default pre trained vgg16 model pre trained in image net should be used
        '''
        # Call super constructor
        super(VGG16, self).__init__()
        # Load model
        if path_to_pre_trained_model is not None:
            self.vgg16 = torch.load(path_to_pre_trained_model)
        else:
            self.vgg16 = torchvision.models.vgg16(pretrained=False)
        # Convert feature module into model list
        self.vgg16.features = nn.ModuleList(list(self.vgg16.features))
        # Convert classifier into module list
        self.vgg16.classifier = nn.ModuleList(list(self.vgg16.classifier))

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        '''
        Forward pass of the model
        :param input: (torch.Tenor) Input tensor of shape (batch size, channels, height, width)
        :return: (List[torch.Tensor]) List of intermediate features in ascending oder w.r.t. the number VGG layer
        '''
        # Adopt grayscale to rgb if needed
        if input.shape[1] == 1:
            output = input.repeat_interleave(3, dim=1)
        else:
            output = input
        # Init list for features
        features = []
        # Feature path
        for layer in self.vgg16.features:
            output = layer(output)
            if isinstance(layer, nn.MaxPool2d):
                features.append(output)
        # Average pool operation
        output = self.vgg16.avgpool(output)
        # Flatten tensor
        output = output.flatten(start_dim=1)
        # Classification path
        for index, layer in enumerate(self.vgg16.classifier):
            output = layer(output)
            if index == 3 or index == 6:
                features.append(output)
        return features


class SelfAttention(nn.Module):
    '''
    Self attention module proposed in: https://arxiv.org/pdf/1805.08318.pdf.
    '''

    def __init__(self, channels: int) -> None:
        '''
        Constructor
        :param channels: (int) Number of channels to be utilized
        '''
        # Call super constructor
        super(SelfAttention, self).__init__()
        # Init convolutions
        self.query_convolution = spectral_norm(
            nn.Conv2d(in_channels=channels, out_channels=channels // 8, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0), bias=False))
        self.key_convolution = spectral_norm(
            nn.Conv2d(in_channels=channels, out_channels=channels // 8, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0), bias=False))
        self.value_convolution = spectral_norm(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0), bias=False))
        # Init gamma parameter
        self.gamma = nn.Parameter(0.1 * torch.ones(1, dtype=torch.float32))

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


class GeneratorResidualBlock(nn.Module):
    '''
    Residual block
    '''

    def __init__(self, in_channels: int, out_channels: int, feature_channels: int,
                 number_of_classes: int = 365) -> None:
        '''
        Constructor
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param number_of_classes: (int) Number of classes in class tensor
        :param feature_channels: (int) Number of feature channels
        '''
        # Call super constructor
        super(GeneratorResidualBlock, self).__init__()
        # Init main operations
        self.main_block = nn.ModuleList([
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ConditionalBatchNorm(num_features=out_channels, number_of_classes=number_of_classes),
            nn.LeakyReLU(negative_slope=0.2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            spectral_norm(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ConditionalBatchNorm(num_features=out_channels, number_of_classes=number_of_classes),
            nn.LeakyReLU(negative_slope=0.2)
        ])
        # Init residual mapping
        self.residual_mapping = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            spectral_norm(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1),
                          padding=(0, 0), bias=False)))
        # Init convolution for mapping the masked features
        self.masked_feature_mapping = spectral_norm(
            nn.Conv2d(in_channels=feature_channels, out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                      bias=False))

    def forward(self, input: torch.Tensor, masked_features: torch.Tensor, class_id: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass
        :param input: (torch.Tensor) Input tensor
        :param masked_features: (torch.Tensor) Masked feature tensor form vvg16
        :param class_id: (torch.Tensor) Class one-hot vector
        :return: (torch.Tensor) Output tensor
        '''
        # Main path
        output = input
        for layer in self.main_block:
            if isinstance(layer, ConditionalBatchNorm):
                output = layer(output, class_id)
            else:
                output = layer(output)
        # Residual mapping
        output_residual = self.residual_mapping(input)
        output_main = output + output_residual
        # Feature path
        mapped_features = self.masked_feature_mapping(masked_features)
        # Addition step
        output = output_main + mapped_features
        return output


class LinearBlock(nn.Module):

    def __init__(self, in_features: int, out_features: int, feature_size: int) -> None:
        '''
        Constructor
        :param in_features: (int) Number of input features
        :param out_features: (int) Number of output features
        :param feature_size: (int) Number of channels including in the feature vector
        '''
        # Call super constructor
        super(LinearBlock, self).__init__()
        # Init linear layer and activation
        self.main_block = nn.Sequential(
            spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=False)),
            nn.LeakyReLU(negative_slope=0.2)
        )
        # Init mapping the masked features
        self.masked_feature_mapping = nn.Linear(in_features=feature_size, out_features=out_features, bias=False)

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


class DiscriminatorResidualBlock(nn.Module):
    '''
    Simple residual block for the discriminator model.
    '''

    def __init__(self, in_channels: int, out_channels: int) -> None:
        '''
        Constructor
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        '''
        # Call super constructor
        super(DiscriminatorResidualBlock, self).__init__()
        # Init operation of the main part
        self.main_block = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1),
                          stride=(1, 1), bias=False)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1),
                          stride=(1, 1), bias=False)),
            nn.LeakyReLU(negative_slope=0.2)
        )
        # Init residual mapping
        self.residual_mapping = spectral_norm(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=(0, 0),
                      stride=(1, 1), bias=False))
        # Init downsmapling
        self.downsampling = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass
        :param input: (torch.Tensor) Input tensor of shape (batch size, in channels, height, width)
        :return: (torch.Tensor) Output tensor of shape (batch size, in channels, height / 2, width / 2)
        '''
        # Main path
        output = self.main_block(input)
        # Residual mapping
        output_residual = self.residual_mapping(input)
        output = output + output_residual
        # Downsampling
        output = self.downsampling(output)
        return output


class ConditionalBatchNorm(nn.Module):
    """
    This class implements conditional batch normalization as used in:
    https://arxiv.org/pdf/1707.00683.pdf
    """

    def __init__(self, num_features: int, number_of_classes: int = 365) -> None:
        """
        Constructor method
        :param num_features: (int) Number of features
        :param number_of_classes: (int) Number of classes in class tensor
        """
        # Call super constructor
        super(ConditionalBatchNorm, self).__init__()
        # Init linear layers to predict the affine parameters
        self.linear_scale = nn.Linear(in_features=number_of_classes, out_features=num_features, bias=False)
        self.linear_bias = nn.Linear(in_features=number_of_classes, out_features=num_features, bias=False)
        # Apply spectral norm
        self.linear_scale = spectral_norm(self.linear_scale)
        self.linear_bias = spectral_norm(self.linear_bias)

    def forward(self, input: torch.Tensor, class_id: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) 4d input feature map
        :param class_id: (torch.Tensor) Class one-hot vector
        :return: (torch.Tensor) Normalized output tensor
        """
        # Perform normalization
        output = (input - input.mean(dim=1, keepdim=True)) / (input.std(dim=1, keepdim=True) + 1e-08)
        # Get affine parameters
        scale = self.linear_scale(class_id)
        bias = self.linear_bias(class_id)
        scale = scale.view(scale.shape[0], scale.shape[-1], 1, 1)
        bias = bias.view(bias.shape[0], bias.shape[-1], 1, 1)
        # Apply affine parameters
        output = scale * output + bias
        return output
