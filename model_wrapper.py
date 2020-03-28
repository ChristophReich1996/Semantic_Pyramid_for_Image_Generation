from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import VGG16, Generator, Discriminator
from lossfunction import SemanticReconstructionLoss, DiversityLoss, LSGANGeneratorLoss, LSGANDiscriminatorLoss


class ModelWrapper(object):
    '''
    Model wrapper implements training, validation and inference of the whole adversarial architecture
    '''

    def __init__(self,
                 generator: Union[Generator, nn.DataParallel],
                 discriminator: Union[Discriminator, nn.DataParallel],
                 training_dataset: DataLoader,
                 vgg16: Union[VGG16, nn.DataParallel] = VGG16(),
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
        self.training_dataset = training_dataset
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
        # Init epoch counter
        epoch_counter = 0
        # Init IS and FID score
        IS, FID = 0, 0
        # Main loop
        while (self.progress_bar.n < training_iterations):
            for images_real, masks in self.training_dataset:
                # Update progress bar with batch size
                self.progress_bar.update(n=images_real.shape[0])
                # Reset gradients
                self.generator.zero_grad()
                self.discriminator.zero_grad()
                # Data to device
                images_real = images_real.to(device)
                for index in range(len(masks)):
                    masks[index].to(device)
                # Get features of images from vgg16 model
                with torch.no_grad():
                    features_real = self.vgg16(images_real)
                # Generate random noise vector
                noise_vector = torch.randn((images_real.shape[0], self.generator.latent_dimensions),
                                           dtype=torch.float32, device=device, requires_grad=True)
                # Generate fake images
                images_fake = self.generator(input=noise_vector, features=features_real, masks=masks)
                # Get discriminator loss
                loss_discriminator_real, loss_discriminator_fake = self.discriminator_loss(images_real, images_fake)
                # Calc gradients
                loss_discriminator_real.backward()
                loss_discriminator_fake.backward()
                # Optimize discriminator
                self.discriminator_optimizer.step()
                # Generate new fake images
                images_fake = self.generator(input=noise_vector, features=features_real, masks=masks)
                # Get generator loss
                loss_generator = self.generator_loss(images_fake)
                # Get diversity loss
                loss_generator_diversity = self.diversity_loss(images_fake, noise_vector)
                # Get features of fake images
                features_fake = self.vgg16(images_fake)
                # Calc semantic reconstruction loss
                loss_generator_semantic_reconstruction = self.semantic_reconstruction_loss(features_real, features_fake)
                # Calc complied loss
                loss_generator_complied = loss_generator + loss_generator_diversity + \
                                          loss_generator_semantic_reconstruction
                # Calc gradients
                loss_generator_complied.backward()
                # Optimize generator
                self.generator_optimizer.step()
                # Show losses in progress bar description
                self.progress_bar.set_description(
                    'IS={:.4f}, FID={:.4f}, Loss G={:.4f}, Loss D={:.4f}'.format(IS, FID, loss_generator.item(), (
                            loss_discriminator_fake + loss_discriminator_real).item()))
            # Validate model
            if epoch_counter % validate_after_n_epochs == 0:
                IS, FID = self.validate()  # IS in upper case cause is is a key word...
            # Increment epoch counter
            epoch_counter += 1
        # Close progress bar
        self.progress_bar.close()

    def validate(self) -> Union[float, float]:
        '''
        IS and FID score gets estimated
        :return: (float, float) IS and FID score
        '''
        return 0.0, 0.0

    def inference(self) -> torch.Tensor:
        pass


# Testing
if __name__ == '__main__':
    import data
    from torch.utils.data import DataLoader

    # Init models, optimizers and dataset
    generator = Generator()
    discriminator = Discriminator()
    generator_optimizer = torch.optim.Adam(generator.parameters())
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters())
    training_dataset = DataLoader(data.PseudoDataset(length=10), batch_size=2,
                                  collate_fn=data.tensor_list_of_masks_collate_function)
    # Init model wrapper
    model_wrapper = ModelWrapper(generator=generator, discriminator=discriminator, training_dataset=training_dataset,
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer)
    # Perform training
    model_wrapper.train(training_iterations=100, device='cpu')
