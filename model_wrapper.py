from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

from models import VGG16, Generator, Discriminator
from lossfunction import SemanticReconstructionLoss, DiversityLoss, LSGANGeneratorLoss, LSGANDiscriminatorLoss
from data import image_label_list_of_masks_collate_function


class ModelWrapper(object):
    '''
    Model wrapper implements training, validation and inference of the whole adversarial architecture
    '''

    def __init__(self,
                 generator: Union[Generator, nn.DataParallel],
                 discriminator: Union[Discriminator, nn.DataParallel],
                 training_dataset: DataLoader,
                 validation_dataset: DataLoader,
                 vgg16: Union[VGG16, nn.DataParallel] = VGG16(),
                 generator_optimizer: torch.optim.Optimizer = None,
                 discriminator_optimizer: torch.optim.Optimizer = None,
                 generator_loss: nn.Module = LSGANGeneratorLoss(),
                 discriminator_loss: nn.Module = LSGANDiscriminatorLoss(),
                 semantic_reconstruction_loss: nn.Module = SemanticReconstructionLoss(),
                 diversity_loss: nn.Module = DiversityLoss(),
                 save_data_path: str = 'saved_data') -> None:
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
        self.validation_dataset = validation_dataset
        self.vgg16 = vgg16
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.semantic_reconstruction_loss = semantic_reconstruction_loss
        self.diversity_loss = diversity_loss
        self.latent_dimensions = self.generator.module.latent_dimensions \
            if isinstance(self.generator, nn.DataParallel) else self.generator.latent_dimensions
        # Make directories to save logs, plots and models during training
        time_and_date = str(datetime.now())
        self.path_save_models = os.path.join(save_data_path, 'models_' + time_and_date)
        if not os.path.exists(self.path_save_models):
            os.makedirs(self.path_save_models)
        self.path_save_plots = os.path.join(save_data_path, 'plots_' + time_and_date)
        if not os.path.exists(self.path_save_plots):
            os.makedirs(self.path_save_plots)
        self.path_save_metrics = os.path.join(save_data_path, 'metrics_' + time_and_date)
        if not os.path.exists(self.path_save_metrics):
            os.makedirs(self.path_save_metrics)
        # Make indexes for validation plots
        validation_plot_indexes = np.random.choice(range(len(self.validation_dataset)), 49, replace=False)
        # Plot and save validation images used to plot generated images
        self.validation_images_to_plot, _, self.validation_masks = image_label_list_of_masks_collate_function(
            [self.validation_dataset.dataset[index] for index in validation_plot_indexes])
        torchvision.utils.save_image(self.validation_images_to_plot, os.path.join(self.path_save_plots,
                                                                                  'validation_images.png'), nrow=7)
        # Plot masks
        torchvision.utils.save_image(self.validation_masks[0],
                                     os.path.join(self.path_save_plots, 'validation_masks.png'),
                                     nrow=7)
        # Generate latents for validation
        self.validation_latents = torch.randn(49, self.latent_dimensions, dtype=torch.float32)

    def train(self, epochs: int = 20, validate_after_n_iterations: int = 10000, device: str = 'cuda',
              save_model_after_n_epochs: int = 10) -> None:
        # Adopt to batch size
        validate_after_n_iterations = (validate_after_n_iterations // self.training_dataset.batch_size) \
                                      * self.training_dataset.batch_size
        # Models into training mode
        self.generator.train()
        self.discriminator.train()
        # Vgg16 into eval mode
        self.vgg16.eval()
        # Models to device
        self.generator.to(device)
        self.discriminator.to(device)
        self.vgg16.to(device)
        # Init progress bar
        self.progress_bar = tqdm(total=epochs * len(self.training_dataset), dynamic_ncols=True)
        # Init epoch counter
        # Init IS and FID score
        IS, FID = 0, 0
        # Main loop
        for epoch in range(epochs):
            for images_real, labels, masks in self.training_dataset:
                # Update progress bar with batch size
                self.progress_bar.update(n=images_real.shape[0])
                # Reset gradients
                self.generator.zero_grad()
                self.discriminator.zero_grad()
                # Data to device
                images_real = images_real.to(device)
                labels = labels.to(device)
                for index in range(len(masks)):
                    masks[index] = masks[index].to(device)
                # Get features of images from vgg16 model
                with torch.no_grad():
                    features_real = self.vgg16(images_real)
                # Generate random noise vector
                noise_vector = torch.randn((images_real.shape[0], self.latent_dimensions),
                                           dtype=torch.float32, device=device, requires_grad=True)
                # Generate fake images
                images_fake = self.generator(input=noise_vector, features=features_real, masks=masks)
                # Discriminator prediction real
                prediction_real = self.discriminator(images_real, labels)
                # Discriminator prediction fake
                prediction_fake = self.discriminator(images_fake, labels)
                # Get discriminator loss
                loss_discriminator_real, loss_discriminator_fake = self.discriminator_loss(prediction_real,
                                                                                           prediction_fake)
                # Calc gradients
                loss_discriminator_real.backward()
                loss_discriminator_fake.backward()
                # Optimize discriminator
                self.discriminator_optimizer.step()
                # Generate new fake images
                images_fake = self.generator(input=noise_vector, features=features_real, masks=masks)
                # Discriminator prediction fake
                prediction_fake = self.discriminator(images_fake, labels)
                # Get generator loss
                loss_generator = self.generator_loss(prediction_fake)
                # Get diversity loss
                loss_generator_diversity = self.diversity_loss(images_fake, noise_vector)
                # Get features of fake images
                features_fake = self.vgg16(images_fake)
                # Calc semantic reconstruction loss
                loss_generator_semantic_reconstruction = \
                    self.semantic_reconstruction_loss(features_real, features_fake, masks)
                # Calc complied loss
                loss_generator_complied = loss_generator + loss_generator_semantic_reconstruction \
                                          + loss_generator_diversity
                # Calc gradients
                loss_generator_complied.backward()
                # Optimize generator
                self.generator_optimizer.step()
                # Show losses in progress bar description
                self.progress_bar.set_description(
                    'Loss Div={:.4f}, Loss Rec={:.4f}, Loss G={:.4f}, Loss D={:.4f}'.format(
                        loss_generator_diversity.item(), loss_generator_semantic_reconstruction.item(),
                        loss_generator.item(), (loss_discriminator_fake + loss_discriminator_real).item()))
                # Validate model
                if self.progress_bar.n % validate_after_n_iterations == 0:
                    print('Val')
                    IS, FID = self.validate(device=device)  # IS in upper case cause is is a key word...
            if epoch % save_model_after_n_epochs == 0:
                torch.save(self.generator, os.path.join(self.path_save_models, 'generator_{}.pt'.format(epoch)))
                torch.save(self.discriminator, os.path.join(self.path_save_models, 'discriminator_{}.pt'.format(epoch)))
        # Close progress bar
        self.progress_bar.close()

    @torch.no_grad()
    def validate(self, plot: bool = True, device: str = 'cuda') -> Union[float, float]:
        '''
        IS and FID score gets estimated
        :param plot: (bool) True if samples should be plotted
        :return: (float, float) IS and FID score
        '''
        # Generator into validation mode
        self.generator.eval()
        self.vgg16.eval()
        # Validation samples for plotting to device
        self.validation_latents = self.validation_latents.to(device)
        self.validation_images_to_plot = self.validation_images_to_plot.to(device)
        for index in range(len(self.validation_masks)):
            self.validation_masks[index] = self.validation_masks[index].to(device)
        # Generate images
        fake_image = self.generator(input=self.validation_latents,
                                    features=self.vgg16(self.validation_images_to_plot),
                                    masks=self.validation_masks).cpu()
        # Save images
        torchvision.utils.save_image(fake_image, os.path.join(self.path_save_plots, str(self.progress_bar.n) + '.png'),
                                     nrow=7)
        # Generator back into train mode
        self.generator.train()
        return 0.0, 0.0

    def inference(self) -> torch.Tensor:
        pass
