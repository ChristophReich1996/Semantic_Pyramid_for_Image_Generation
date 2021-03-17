from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from tqdm import tqdm
import numpy as np
from datetime import datetime
import os

from models import VGG16, Generator, Discriminator
from lossfunction import SemanticReconstructionLoss, DiversityLoss, LSGANGeneratorLoss, LSGANDiscriminatorLoss
from data import image_label_list_of_masks_collate_function
from frechet_inception_distance import frechet_inception_distance
from misc import Logger, get_masks_for_inference
import misc


class ModelWrapper(object):
    '''
    Model wrapper implements training, validation and inference of the whole adversarial architecture
    '''

    def __init__(self,
                 generator: Union[Generator, nn.DataParallel],
                 discriminator: Union[Discriminator, nn.DataParallel],
                 training_dataset: DataLoader,
                 validation_dataset: Dataset,
                 validation_dataset_fid: DataLoader,
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
        self.validation_dataset_fid = validation_dataset_fid
        self.vgg16 = vgg16
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.semantic_reconstruction_loss = semantic_reconstruction_loss
        self.diversity_loss = diversity_loss
        self.latent_dimensions = self.generator.module.latent_dimensions \
            if isinstance(self.generator, nn.DataParallel) else self.generator.latent_dimensions
        # Init logger
        self.logger = Logger()
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
        validation_plot_indexes = np.random.choice(range(len(self.validation_dataset_fid.dataset)), 49, replace=False)
        # Plot and save validation images used to plot generated images
        self.validation_images_to_plot, self.validation_labels, self.validation_masks \
            = image_label_list_of_masks_collate_function(
            [self.validation_dataset_fid.dataset[index] for index in validation_plot_indexes])

        torchvision.utils.save_image(misc.normalize_0_1_batch(self.validation_images_to_plot),
                                     os.path.join(self.path_save_plots, 'validation_images.png'), nrow=7)
        # Plot masks
        torchvision.utils.save_image(self.validation_masks[0],
                                     os.path.join(self.path_save_plots, 'validation_masks.png'),
                                     nrow=7)
        # Generate latents for validation
        self.validation_latents = torch.randn(49, self.latent_dimensions, dtype=torch.float32)
        # Log hyperparameter
        self.logger.hyperparameter['generator'] = str(self.generator)
        self.logger.hyperparameter['discriminator'] = str(self.discriminator)
        self.logger.hyperparameter['vgg16'] = str(self.vgg16)
        self.logger.hyperparameter['generator_optimizer'] = str(self.generator_optimizer)
        self.logger.hyperparameter['discriminator_optimizer'] = str(self.discriminator_optimizer)
        self.logger.hyperparameter['generator_loss'] = str(self.generator_loss)
        self.logger.hyperparameter['discriminator_loss'] = str(self.discriminator_loss)
        self.logger.hyperparameter['diversity_loss'] = str(self.diversity_loss)
        self.logger.hyperparameter['discriminator_loss'] = str(self.semantic_reconstruction_loss)

    def train(self, epochs: int = 20, validate_after_n_iterations: int = 100000, device: str = 'cuda',
              save_model_after_n_epochs: int = 1, w_rec: float = 0.1, w_div: float = 0.1) -> None:
        """
        Training method
        :param epochs: (int) Number of epochs to perform
        :param validate_after_n_iterations: (int) Number of iterations after the model gets validated
        :param device: (str) Device to be used
        :param save_model_after_n_epochs: (int) Epochs to perform after model gets saved
        :param w_rec: (float) Weight factor for the reconstruction loss
        :param w_div: (float) Weight factor for the diversity loss
        """
        # Save weights factors
        self.logger.hyperparameter['w_rec'] = str(w_rec)
        self.logger.hyperparameter['w_div'] = str(w_div)
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
        self.progress_bar = tqdm(total=epochs * len(self.training_dataset.dataset), dynamic_ncols=True)
        # Initial validation
        self.progress_bar.set_description('Validation')
        self.inference(device=device)
        fid = self.validate(device=device)
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
                    images_fake = self.generator(input=noise_vector, features=features_real, masks=masks,
                                                 class_id=labels)
                # Discriminator prediction real
                prediction_real = self.discriminator(images_real, labels)
                # Discriminator prediction fake
                prediction_fake = self.discriminator(images_fake, labels)
                # Get discriminator loss
                loss_discriminator_real, loss_discriminator_fake = self.discriminator_loss(prediction_real,
                                                                                           prediction_fake)
                # Calc gradients
                (loss_discriminator_real + loss_discriminator_fake).backward()
                # Optimize discriminator
                self.discriminator_optimizer.step()
                # Reset gradients of generator and discriminator
                self.generator.zero_grad()
                self.discriminator.zero_grad()
                # Generate new fake images
                images_fake = self.generator(input=noise_vector, features=features_real, masks=masks, class_id=labels)
                # Discriminator prediction fake
                prediction_fake = self.discriminator(images_fake, labels)
                # Get generator loss
                loss_generator = self.generator_loss(prediction_fake)
                # Get diversity loss
                loss_generator_diversity = w_div * self.diversity_loss(images_fake, noise_vector)
                # Get features of fake images
                features_fake = self.vgg16(images_fake)
                # Calc semantic reconstruction loss
                loss_generator_semantic_reconstruction = \
                    w_rec * self.semantic_reconstruction_loss(features_real, features_fake, masks)
                # Calc complied loss
                loss_generator_complied = loss_generator + loss_generator_semantic_reconstruction \
                                          + loss_generator_diversity
                # Calc gradients
                loss_generator_complied.backward()
                # Optimize generator
                self.generator_optimizer.step()
                # Show losses in progress bar description
                self.progress_bar.set_description(
                    'FID={:.4f}, Loss Div={:.4f}, Loss Rec={:.4f}, Loss G={:.4f}, Loss D={:.4f}'.format(
                        fid, loss_generator_diversity.item(), loss_generator_semantic_reconstruction.item(),
                        loss_generator.item(), (loss_discriminator_fake + loss_discriminator_real).item()))
                # Log losses
                self.logger.log(metric_name='loss_discriminator_real', value=loss_discriminator_real.item())
                self.logger.log(metric_name='loss_discriminator_fake', value=loss_discriminator_fake.item())
                self.logger.log(metric_name='loss_generator', value=loss_generator.item())
                self.logger.log(metric_name='loss_generator_semantic_reconstruction',
                                value=loss_generator_semantic_reconstruction.item())
                self.logger.log(metric_name='loss_generator_diversity', value=loss_generator_diversity.item())
                self.logger.log(metric_name='iterations', value=self.progress_bar.n)
                self.logger.log(metric_name='epoch', value=epoch)
                # Validate model
                if self.progress_bar.n % validate_after_n_iterations == 0:
                    self.progress_bar.set_description('Validation')
                    fid = self.validate(device=device)
                    self.inference(device=device)
                    # Log fid
                    self.logger.log(metric_name='fid', value=fid)
                    self.logger.log(metric_name='iterations_fid', value=self.progress_bar.n)
                    # Save all logs
                    self.logger.save_metrics(self.path_save_metrics)
            if epoch % save_model_after_n_epochs == 0:
                torch.save(self.generator, os.path.join(self.path_save_models, 'generator_{}.pt'.format(epoch)))
                torch.save(self.discriminator, os.path.join(self.path_save_models, 'discriminator_{}.pt'.format(epoch)))
            self.inference(device=device)
            # Save all logs
            self.logger.save_metrics(self.path_save_metrics)
        # Close progress bar
        self.progress_bar.close()

    @torch.no_grad()
    def validate(self, device: str = 'cuda') -> float:
        '''
        FID score gets estimated
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
                                    masks=self.validation_masks,
                                    class_id=self.validation_labels).cpu()
        # Save images
        torchvision.utils.save_image(misc.normalize_0_1_batch(fake_image),
                                     os.path.join(self.path_save_plots, str(self.progress_bar.n) + '.png'),
                                     nrow=7)
        # Generator back into train mode
        self.generator.train()
        return frechet_inception_distance(dataset_real=self.validation_dataset_fid,
                                          generator=self.generator, vgg16=self.vgg16)

    @torch.no_grad()
    def inference(self, device: str = 'cuda') -> None:
        '''
        Random images for different feature levels are generated and saved
        '''
        # Models to device
        self.generator.to(device)
        self.vgg16.to(device)
        # Generator into eval mode
        self.generator.eval()
        # Get random images form validation dataset
        images = [self.validation_dataset[index].unsqueeze(dim=0).to(device) for index in
                  np.random.choice(range(len(self.validation_dataset)), replace=False, size=7)]
        # Get list of masks for different layers
        masks_levels = [get_masks_for_inference(layer, add_batch_size=True, device=device) for layer in range(7)]
        # Init tensor of fake images to store all fake images
        fake_images = torch.empty(7 ** 2, images[0].shape[1], images[0].shape[2], images[0].shape[3],
                                  dtype=torch.float32, device=device)
        # Init counter
        counter = 0
        # Loop over all image and masks
        for image, label, _ in images:
            for masks in masks_levels:
                # Generate fake images
                fake_image = self.generator(
                    input=torch.randn(1, self.latent_dimensions, dtype=torch.float32, device=device),
                    features=self.vgg16(image),
                    masks=masks,
                    class_id=label)
                # Save fake images
                fake_images[counter] = fake_image.squeeze(dim=0)
                # Increment counter
                counter += 1
        # Save tensor as image
        torchvision.utils.save_image(
            misc.normalize_0_1_batch(fake_images),
            os.path.join(self.path_save_plots, 'predictions_{}.png'.format(str(datetime.now()))), nrow=7)
