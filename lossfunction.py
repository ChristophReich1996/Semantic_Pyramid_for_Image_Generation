from typing import List, Tuple

import torch
import torch.nn as nn
import kornia


class SemanticReconstructionLoss(nn.Module):
    '''
    Implementation of the proposed semantic reconstruction loss
    '''

    def __init__(self) -> None:
        '''
        Constructor
        '''
        # Call super constructor
        super(SemanticReconstructionLoss, self).__init__()
        # Init max pooling operations. Since the features have various dimensions, 2d & 1d max pool as the be init
        self.max_pooling_2d = nn.MaxPool2d(2)
        self.max_pooling_1d = nn.MaxPool1d(2)

    def __repr__(self):
        '''
        Get representation of the loss module
        :return: (str) String including information
        '''
        return '{}, maxpool kernel size{}' \
            .format(self.__class__.__name__, self.max_pooling_1d.kernel_size)

    def forward(self, features_real: List[torch.Tensor], features_fake: List[torch.Tensor],
                masks: List[torch.Tensor]) -> torch.Tensor:
        '''
        Forward pass
        :param features_real: (List[torch.Tensor]) List of real features
        :param features_fake: (List[torch.Tensor]) List of fake features
        :return: (torch.Tensor) Loss
        '''
        # Check lengths
        assert len(features_real) == len(features_fake) == len(masks)
        # Init loss
        loss = torch.tensor([0.0], dtype=torch.float32, device=features_real[0].device)
        # Calc full loss
        for feature_real, feature_fake, mask in zip(features_real, features_fake, masks):
            # Downscale features
            if len(feature_fake.shape) == 4:
                feature_real = self.max_pooling_2d(feature_real)
                feature_fake = self.max_pooling_2d(feature_fake)
                mask = self.max_pooling_2d(mask)
                # Normalize features
                """
                union = torch.cat((feature_real, feature_fake), dim=0)
                feature_real, feature_fake = \
                    kornia.normalize_min_max(union).split(split_size=feature_fake.shape[0], dim=0)
                """
            else:
                feature_real = self.max_pooling_1d(feature_real.unsqueeze(dim=1))
                feature_fake = self.max_pooling_1d(feature_fake.unsqueeze(dim=1))
                mask = self.max_pooling_1d(mask.unsqueeze(dim=1))
                # Normalize features
                """
                union = torch.cat((feature_real, feature_fake), dim=0)
                feature_real, feature_fake = \
                    kornia.normalize_min_max(union.unsqueeze(dim=1)).split(split_size=feature_fake.shape[0], dim=0)
                """
            # Calc l1 loss of the real and fake feature conditionalized by the corresponding mask
            loss = loss + torch.mean(torch.abs((feature_real - feature_fake) * mask))
        return loss


class DiversityLoss(nn.Module):
    '''
    Implementation of the mini-batch diversity loss
    '''

    def __init__(self) -> None:
        '''
        Constructor
        '''
        # Call super constructor
        super(DiversityLoss, self).__init__()
        # Init l1 loss module
        self.l1_loss = nn.L1Loss(reduction='mean')

    def __repr__(self):
        '''
        Get representation of the loss module
        :return: (str) String including information
        '''
        return self.__class__.__name__

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
        loss = self.l1_loss(latent_inputs_1, latent_inputs_2) \
               / ( self.l1_loss(images_fake_1, images_fake_2) + 1e-08)
        return loss




class LSGANGeneratorLoss(nn.Module):
    '''
    Implementation of the least squares gan loss for the generator network
    '''

    def __init__(self) -> None:
        # Call super constructor
        super(LSGANGeneratorLoss, self).__init__()

    def __repr__(self):
        '''
        Get representation of the loss module
        :return: (str) String including information
        '''
        return str(self.__class__.__name__)

    def forward(self, prediction_fake: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass
        :param prediction_fake: (torch.Tensor) Fake images generated
        :return: (torch.Tensor) Loss
        '''
        return 0.5 * torch.mean((prediction_fake - 1.0) ** 2)


class LSGANDiscriminatorLoss(nn.Module):
    '''
    Implementation of the least squares gan loss for the discriminator network
    '''

    def __init__(self) -> None:
        # Call super constructor
        super(LSGANDiscriminatorLoss, self).__init__()

    def __repr__(self):
        '''
        Get representation of the loss module
        :return: (str) String including information
        '''
        return str(self.__class__.__name__)

    def forward(self, prediction_real: torch.Tensor, prediction_fake: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        '''
        Forward pass. Loss parts are not summed up to not retain the whole backward graph later.
        :param prediction_real: (torch.Tensor) Real images
        :param prediction_fake: (torch.Tensor) Fake images generated
        :return: (torch.Tensor) Loss real part and loss fake part
        '''
        return 0.5 * torch.mean((prediction_real - 1.0) ** 2), 0.5 * torch.mean(prediction_fake ** 2)
