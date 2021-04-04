from typing import List, Union, Optional

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import kornia
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
        self.linear_layer = spectral_norm(
            nn.Linear(in_features=latent_dimensions, out_features=latent_dimensions, bias=True))
        self.linear_block_1 = LinearBlock(in_features=latent_dimensions, out_features=365, feature_size=365)
        self.linear_block_2 = LinearBlock(in_features=365, out_features=2048, feature_size=4096)
        self.convolution_layer = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=int(512 // channels_factor), kernel_size=(1, 1),
                                    padding=(0, 0), stride=(1, 1), bias=True)))
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
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.BatchNorm2d(int(64 // channels_factor)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(in_channels=int(64 // channels_factor), out_channels=int(64 // channels_factor),
                                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(
                nn.Conv2d(in_channels=int(64 // channels_factor), out_channels=out_channels, kernel_size=(1, 1),
                          stride=(1, 1), padding=(0, 0), bias=True))
        )
        # Init weights
        self.apply(init_weights)

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
        # Initial linear layer
        output = self.linear_layer(input)
        # Forward pass linear blocks
        output = self.linear_block_1(output, features[depth_counter] * masks[depth_counter])
        depth_counter -= 1
        output = self.linear_block_2(output, features[depth_counter] * masks[depth_counter])
        depth_counter -= 1
        # Reshaping
        output = output.view(output.shape[0], -1, 4, 4)
        # Forward pass last linear layer
        output = self.convolution_layer(output)
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
        return output.tanh()


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
            DiscriminatorInputResidualBlock(in_channels=in_channels, out_channels=int(64 // channel_factor)),
            DiscriminatorResidualBlock(in_channels=int(64 // channel_factor), out_channels=int(128 // channel_factor)),
            DiscriminatorResidualBlock(in_channels=int(128 // channel_factor), out_channels=int(256 // channel_factor)),
            SelfAttention(channels=int(256 // channel_factor)),
            DiscriminatorResidualBlock(in_channels=int(256 // channel_factor), out_channels=int(256 // channel_factor)),
            DiscriminatorResidualBlock(in_channels=int(256 // channel_factor), out_channels=int(256 // channel_factor)),
            DiscriminatorResidualBlock(in_channels=int(256 // channel_factor), out_channels=int(512 // channel_factor)),
            DiscriminatorResidualBlock(in_channels=int(512 // channel_factor), out_channels=int(768 // channel_factor)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            spectral_norm(nn.Linear(in_features=int(768 // channel_factor), out_features=128, bias=True)),
            nn.LeakyReLU(negative_slope=0.2)
        )
        # Init classification layer
        self.classification = spectral_norm(
            nn.Linear(in_features=128, out_features=1, bias=True))
        # Init embedding layer
        self.embedding = spectral_norm(nn.Embedding(num_embeddings=number_of_classes,
                                                    embedding_dim=128))
        # Init weights
        self.apply(init_weights)

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
        output_embedding = self.embedding(class_id.argmax(dim=-1, keepdim=True))
        output_embedding = output * output_embedding
        # Classification path
        output = self.classification(output)
        return output + output_embedding


class VGG16(nn.Module):
    '''
    Implementation of a pre-trained VGG 16 model which outputs intermediate feature activations of the model.
    '''

    def __init__(self, path_to_pre_trained_model: Optional[str] = None, return_output: Optional[bool] = False) -> None:
        '''
        Constructor
        :param pretrained: (bool) True if the default pre trained vgg16 model pre trained in image net should be used
        '''
        # Call super constructor
        super(VGG16, self).__init__()
        # Save parameter
        self.return_output = return_output
        # Load model
        if path_to_pre_trained_model is not None:
            self.vgg16 = torch.load(path_to_pre_trained_model)
        else:
            self.vgg16 = torchvision.models.vgg16(pretrained=False)
            self.vgg16.classifier[-1] = nn.Linear(in_features=4096, out_features=365, bias=True)
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
        # Normalize input
        output = kornia.normalize(output,
                                  mean=torch.tensor([0.485, 0.456, 0.406], device=output.device),
                                  std=torch.tensor([0.229, 0.224, 0.225], device=output.device))
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
        if self.return_output:
            return output
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
                      padding=(0, 0), bias=True))
        self.key_convolution = spectral_norm(
            nn.Conv2d(in_channels=channels, out_channels=channels // 8, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0), bias=True))
        self.value_convolution = spectral_norm(
            nn.Conv2d(in_channels=channels, out_channels=channels // 2, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0), bias=True))
        self.attention_convolution = spectral_norm(
            nn.Conv2d(in_channels=channels // 2, out_channels=channels, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0), bias=True))
        # Init max pooling
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
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
        query = self.query_convolution(input)
        key = self.key_convolution(self.max_pooling(input))
        value = self.value_convolution(self.max_pooling(input))
        # Reshape key, query and value mapping
        query = query.view(-1, channels // 8, height * width).permute(0, 2, 1)
        key = key.view(-1, channels // 8, height * width // 4)
        value = value.view(-1, channels // 2, height * width // 4)
        # Calc attention map
        attention_map = torch.bmm(query, key).softmax(dim=-1).permute(0, 2, 1)
        # Apply attention map to value to obtain the attention output features
        attention_features = torch.bmm(value, attention_map)
        # Reshape attention features
        attention_features = attention_features.view(-1, channels // 2, height, width)
        # Get output features
        output = self.attention_convolution(attention_features)
        # Residual mapping and gamma multiplication
        output = self.gamma * output + input
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
            ConditionalBatchNorm(num_features=in_channels, number_of_classes=number_of_classes),
            nn.LeakyReLU(negative_slope=0.2),
            nn.UpsamplingNearest2d(scale_factor=2),
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)),
            ConditionalBatchNorm(num_features=out_channels, number_of_classes=number_of_classes),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True))
        ])
        # Init residual mapping
        self.residual_mapping = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                    stride=(1, 1), padding=(0, 0), bias=True)))
        # Init convolution for mapping the masked features
        self.masked_feature_mapping = spectral_norm(
            nn.Conv2d(in_channels=feature_channels, out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                      bias=True))

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
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=True))
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


class DiscriminatorInputResidualBlock(nn.Module):
    '''
    Simple input residual block for the discriminator model.
    '''

    def __init__(self, in_channels: int, out_channels: int) -> None:
        '''
        Constructor
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        '''
        # Call super constructor
        super(DiscriminatorInputResidualBlock, self).__init__()
        # Init operation of the main part
        self.main_block = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1),
                          stride=(1, 1), bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1),
                          stride=(1, 1), bias=True))
        )
        # Init residual mapping
        self.residual_mapping = spectral_norm(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=(0, 0),
                      stride=(1, 1), bias=True))
        # Init downsmapling
        self.downsampling = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass
        :param input: (torch.Tensor) Input tensor of shape (batch size, in channels, height, width)
        :return: (torch.Tensor) Output tensor of shape (batch size, in channels, height / 2, width / 2)
        '''
        # Main path
        output = self.downsampling(self.main_block(input))
        # Residual mapping
        output_residual = self.residual_mapping(self.downsampling(input))
        output = output + output_residual
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
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1),
                          stride=(1, 1), bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1),
                          stride=(1, 1), bias=True))
        )
        # Init residual mapping
        self.residual_mapping = spectral_norm(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=(0, 0),
                      stride=(1, 1), bias=True))
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
        # Init batch normalization layer without affine parameters
        self.batch_norm = nn.BatchNorm2d(num_features=num_features, momentum=0.001, affine=False)
        # Init embedding layer for producing the affine parameters
        self.embedding = nn.Embedding(num_embeddings=number_of_classes, embedding_dim=num_features * 2)
        # Init weight matrix of embedding layer
        self.embedding.weight.data[:, :num_features].fill_(1.)
        self.embedding.weight.data[:, num_features:].zero_()

    def forward(self, input: torch.Tensor, class_id: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) 4d input feature map
        :param class_id: (torch.Tensor) Class one-hot vector
        :return: (torch.Tensor) Normalized output tensor
        """
        # Perform normalization
        output = self.batch_norm(input)
        # Get affine parameters
        scale, bias = self.embedding(class_id.argmax(dim=-1, keepdim=False)).chunk(chunks=2, dim=1)
        scale = scale.view(scale.shape[0], scale.shape[-1], 1, 1)
        bias = bias.view(bias.shape[0], bias.shape[-1], 1, 1)
        # Apply affine parameters
        output = scale * output + bias
        return output


def init_weights(module: nn.Module) -> None:
    """
    Init weights of linear and convolution layers
    :param module: (nn.Module) Module
    """
    # Check if module is a linear or a convolution layer
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        # Init weights
        nn.init.xavier_uniform_(module.weight)
        # Init bias
        module.bias.data.fill_(0.)
