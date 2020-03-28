from typing import List, Tuple

import torch
import torch.nn as nn


class SemanticReconstructionLoss(nn.Module):
    '''
    Implementation of the proposed semantic reconstruction loss
    '''

    def __init__(self, weight_factor: float = 0.1) -> None:
        '''
        Constructor
        '''
        # Call super constructor
        super(SemanticReconstructionLoss, self).__init__()
        # Save parameter
        self.weight_factor = weight_factor
        # Init l1 loss module
        self.l1_loss = nn.L1Loss(reduction='mean')
        # Init max pooling operations. Since the features have various dimensions, 2d & 1d max pool as the be init
        self.max_pooling_2d = nn.MaxPool2d(2)
        self.max_pooling_1d = nn.MaxPool1d(2)

    def forward(self, features_real: List[torch.Tensor], features_fake: List[torch.Tensor]) -> torch.Tensor:
        '''
        Forward pass
        :param features_real: (List[torch.Tensor]) List of real features
        :param features_fake: (List[torch.Tensor]) List of fake features
        :return: (torch.Tensor) Loss
        '''
        # Init loss
        loss = torch.tensor([0.0], dtype=torch.float32, device=features_real[0].device)
        # Calc full loss
        for feature_real, feature_fake in zip(features_real, features_fake):
            # Downscale features
            if len(feature_fake.shape) == 4:
                feature_real = self.max_pooling_2d(feature_real)
                feature_fake = self.max_pooling_2d(feature_fake)
            else:
                feature_real = self.max_pooling_1d(feature_real.unsqueeze(dim=1))
                feature_fake = self.max_pooling_1d(feature_fake.unsqueeze(dim=1))
            # Normalize features
            feature_real = (feature_real - feature_real.mean()) / feature_real.std()
            feature_fake = (feature_fake - feature_fake.mean()) / feature_fake.std()
            # Calc l1 loss of the real and fake feature
            loss = loss + self.l1_loss(feature_real, feature_fake)
        # Average loss with number of features
        loss = loss / len(features_real)
        return self.weight_factor * loss


class DiversityLoss(nn.Module):
    '''
    Implementation of the mini-batch diversity loss
    '''

    def __init__(self, weight_factor: float = 0.1) -> None:
        '''
        Constructor
        '''
        # Call super constructor
        super(DiversityLoss, self).__init__()
        # Save parameter
        self.weight_factor = weight_factor
        # Init l1 loss module
        self.l1_loss = nn.L1Loss(reduction='mean')
        # Init epsilon for numeric stability
        self.epsilon = 1e-08

    def forward(self, images_fake: torch.Tensor, latent_inputs: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass
        :param images_real: (torch.Tensor) Mini-batch of real images
        :param latent_inputs: (torch.Tensor) Random latent input tensor
        :return: (torch.Tensor) Loss
        '''
        # Check batch sizes
        assert images_fake.shape[0] > 1
        # Divide mini-batch of images into two paris
        images_fake_1 = images_fake[:images_fake.shape[0] // 2]
        images_fake_2 = images_fake[images_fake.shape[0] // 2:]
        # Divide latent inputs into two paris
        latent_inputs_1 = latent_inputs[:latent_inputs.shape[0] // 2]
        latent_inputs_2 = latent_inputs[latent_inputs.shape[0] // 2:]
        # Calc loss
        loss = self.l1_loss(latent_inputs_1, latent_inputs_2) / (
                self.l1_loss(images_fake_1, images_fake_2) + self.epsilon)
        return self.weight_factor * loss


class LSGANGeneratorLoss(nn.Module):
    '''
    Implementation of the least squares gan loss for the generator network
    '''

    def __init__(self) -> None:
        # Call super constructor
        super(LSGANGeneratorLoss, self).__init__()

    def forward(self, images_fake: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass
        :param images_fake: (torch.Tensor) Fake images generated
        :return: (torch.Tensor) Loss
        '''
        return 0.5 * torch.mean((images_fake - 1.0) ** 2)


class LSGANDiscriminatorLoss(nn.Module):
    '''
    Implementation of the least squares gan loss for the discriminator network
    '''

    def __init__(self) -> None:
        # Call super constructor
        super(LSGANDiscriminatorLoss, self).__init__()

    def forward(self, images_real: torch.Tensor, images_fake: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass. Loss parts are not summed up to not retain the whole backward graph later.
        :param images_real: (torch.Tensor) Real images
        :param images_fake: (torch.Tensor) Fake images generated
        :return: (torch.Tensor) Loss real part and loss fake part
        '''
        return 0.5 * torch.mean((images_real - 1) ** 2), 0.5 * torch.mean(images_fake ** 2)
