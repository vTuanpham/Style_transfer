import random
import sys
import os
sys.path.insert(0,r'./') #Add root directory here

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models
import torchvision.transforms as transforms

import tqdm
from PIL import Image

from src.models.losses import style_loss, total_variation_loss, GramMatrix
from src.models.losses import mse_loss as content_loss

from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.utils.image_plot import plot_image


class Trainer:
    def __init__(self,
                 dataloaders,
                 output_dir: str,
                 lr_scheduler_type,
                 resume_from_checkpoint,
                 seed,
                 with_tracking,
                 report_to,
                 num_train_epochs,
                 weight_decay,
                 per_device_batch_size,
                 gradient_accumulation_steps,
                 do_eval_per_epoch,
                 learning_rate,
                 vgg_model_type: str,
                 alpha: float,
                 beta: float,
                 gamma: float
                 ):

        self.dataloaders = dataloaders.__call__()
        self.learning_rate = learning_rate
        self.vgg_model_type = vgg_model_type
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_train_epochs = num_train_epochs
        self.output_dir = output_dir

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def get_feature_extractor(selected_indices, vgg_model_type="16", device='cpu'):
        if vgg_model_type == "16":
            vgg = models.vgg19(pretrained=True).features
        else:
            vgg = models.vgg16(pretrained=True).features

        layers = list(vgg.children())

        feature_extractors = []
        if isinstance(selected_indices, list):
            for idx in selected_indices:
                if idx < len(layers):
                    feature_extractor = nn.Sequential(*layers[:idx + 1]).eval().to(device)
                    feature_extractors.append(feature_extractor)
                else:
                    raise ValueError(f"Invalid layer index: {idx}. Index should be less than {len(layers)}.")
        else:
            raise ValueError("Selected indices should be provided as a list of layer indices.")

        return feature_extractors

    def build_model(self):
        generator = Generator(num_res_block=25).to(self.device)
        discriminator = Discriminator().to(self.device)
        content_extractors = self.get_feature_extractor([25, 30, 34], device=self.device)
        style_extractors = self.get_feature_extractor([15, 25, 28, 34, 36], device=self.device)

        return generator, discriminator, content_extractors, style_extractors

    def compute_loss(self, content_features, style_features, stylized_output):

        # Compute style loss
        losses = []
        for feature in style_features:
            loss = style_loss(feature['orginal'], feature['model_output'])
            losses.append(loss)
        loss_style = sum(losses) / len(losses)

        # Compute content loss
        losses = []
        for feature in content_features:
            loss = content_loss(feature['orginal'], feature['model_output'])
            losses.append(loss)
        loss_content = sum(losses) / len(losses)

        # Compute variation loss
        variation_loss = total_variation_loss(stylized_output)

        # Compute total loss
        total_loss = self.alpha * loss_content \
                     + self.beta * loss_style + \
                     self.gamma * variation_loss

        return loss_content, loss_style, total_loss

    def train(self):
        img_generator, discriminator, content_extractors, style_extractors = self.build_model()
        # Define the optimizer
        generator_optimizer  = optim.Adam(img_generator.parameters(), lr=self.learning_rate)
        discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        scheduler = lr_scheduler.LinearLR(generator_optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

        img_generator.train()

        # Training loop
        loss_list = []
        for epoch in tqdm.tqdm(range(self.num_train_epochs)):
            for step, batch in enumerate(self.dataloaders):
                content_imgs = batch['content_image'].to(self.device)
                style_imgs = batch['style_image'].to(self.device)

                # Stack content and style in the channels dim so the generator have
                # some context of what to style the content on
                stacked_images = torch.cat((content_imgs, style_imgs), dim=1)

                # Generate stylized output
                stylized_output = img_generator(stacked_images)

                # Extract features from VGG19 for content and style images
                content_features = []
                for feature_extractor in content_extractors:
                    content_features.append({"orginal": feature_extractor(content_imgs),
                                             "model_output": feature_extractor(stylized_output)})

                style_features = []
                for feature_extractor in style_extractors:
                    style_features.append({"orginal": feature_extractor(style_imgs),
                                             "model_output": feature_extractor(stylized_output)})

                loss_content, loss_style, total_loss = self.compute_loss(content_features,
                                                                         style_features, stylized_output)

                # Backpropagation and optimization
                generator_optimizer.zero_grad()
                total_loss.backward()
                generator_optimizer.step()
                scheduler.step()

            loss_list.append(float(total_loss.item()))
            # Print the loss for monitoring
            print(f"Epoch [{epoch + 1}/{self.num_train_epochs}], Total Loss: {total_loss.item()}")
            print(f"Min loss: {min(loss_list)}")
            if float(total_loss.item()) == min(loss_list):
                print(f"Saving epoch [{epoch + 1}/{self.num_train_epochs}]")
                self.save(img_generator)
            else:
                print(f"Discarding epoch [{epoch + 1}/{self.num_train_epochs}]")
                continue

    def save(self, generator, discriminator=None):
        model_path = os.path.join(self.output_dir, "generator" + ".pth")
        torch.save(generator.state_dict(), model_path)
        if discriminator is not None:
            model_path = os.path.join(self.output_dir, "discriminator" + ".pth")
            torch.save(discriminator.state_dict(), model_path)



