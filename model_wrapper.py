from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import VGG16
from lossfunction import SemanticReconstructionLoss, DiversityLoss, LSGANGeneratorLoss, LSGANDiscriminatorLoss


class ModelWrapper(object):
    '''
    Model wrapper implements training, validation and inference of the whole adversarial architecture
    '''

    def __init__(self,
                 generator: Union[nn.Module, nn.DataParallel],
                 discriminator: Union[nn.Module, nn.DataParallel],
                 training_dataset: DataLoader,
                 vgg16: Union[nn.Module, nn.DataParallel] = VGG16(),
                 generator_optimizer: torch.optim.Optimizer = None,
                 discriminator_optimizer: torch.optim.Optimizer = None,
                 generator_loss: nn.Module = LSGANGeneratorLoss(),
                 discriminator_loss: nn.Module = LSGANDiscriminatorLoss(),
                 semantic_reconstruction_loss: nn.Module = SemanticReconstructionLoss(),
                 diversity_loss: nn.Module = DiversityLoss()) -> None:
        '''
        Constructor
        :param generator: (nn.Module, nn.DataParallel) Generator network
        :param discriminator: (nn.Module, nn.DataParallel) Discriminator network
        :param training_dataset: (DataLoader) Training dataset
        :param vgg16: (nn.Module, nn.DataParallel) VGG16 module
        :param generator_optimizer: (torch.optim.Optimizer) Optimizer of the generator network
        :param discriminator_optimizer: (torch.optim.Optimizer) Optimizer of the discriminator network
        :param generator_loss: (nn.Module) Generator loss function
        :param discriminator_loss: (nn.Module) Discriminator loss function
        :param semantic_reconstruction_loss: (nn.Module) Semantic reconstruction loss function
        :param diversity_loss: (nn.Module) Diversity loss function
        '''
        # Save parameters
        self.generator = generator
        self.discriminator = discriminator
        self.vgg16 = vgg16
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.semantic_reconstruction_loss = semantic_reconstruction_loss
        self.diversity_loss = diversity_loss

    def train(self, training_iterations: int = 1000000, validate_after_n_epochs: int = 1, device: str = 'cuda') -> None:
        # Models into training mode
        self.generator.train()
        self.discriminator.train()
        self.vgg16.train()
        # Models to device
        self.generator.to(device)
        self.discriminator.to(device)
        self.vgg16.to(device)
        # Init progress bar
        self.progress_bar = tqdm(total=training_iterations)

    def validate(self) -> Union[float, float]:
        '''
        IS and FID score gets estimated
        :return: (float, float) IS and FID score
        '''
        pass

    def inference(self) -> torch.Tensor:
        pass
