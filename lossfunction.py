from typing import List

import torch
import torch.nn as nn


class SemanticReconstructionLoss(nn.Module):
    '''
    Implementation of the proposed semantic reconstruction loss
    '''


    def __init__(self):
        '''
        Constructor
        '''
        # Call super constructor
        super(SemanticReconstructionLoss, self).__init__()
        # Init l1 loss module
        self.l1_loss = nn.L1Loss(reduction='mean')

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
            # Normalize features
            feature_real = (feature_real - feature_real.mean()) / feature_real.std()
            feature_fake = (feature_fake - feature_fake.mean()) / feature_fake.std()
            # Calc l1 loss of the real and fake feature
            loss = loss + self.l1_loss(feature_real, feature_fake)
        # Average loss with number of features
        loss = loss / len(features_real)
        return loss
