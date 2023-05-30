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

    def get_feature_extractor(self):
        if self.vgg_model_type == '19':
            # Define the VGG19-based feature extractor
            vgg = models.vgg19(pretrained=True).features
            vgg = nn.Sequential(*list(vgg.children())[:36]).eval()
        if self.vgg_model_type == '16':
            # Define the VGG16-based feature extractor
            vgg = models.vgg16(pretrained=True).features
            vgg = nn.Sequential(*list(vgg.children())[:36]).eval()

        return vgg

    def build_model(self):
        generator = Generator(num_residual=5)
        vgg = self.get_feature_extractor()

        return generator, vgg

    def compute_loss(self, content_features, style_features, stylized_features, stylized_output):
        # Compute content loss
        loss_content = content_loss(stylized_features, content_features)
        # Compute style loss
        loss_style = style_loss(stylized_features, style_features)

        # Compute variation loss
        variation_loss = total_variation_loss(stylized_output)

        # Compute total loss
        total_loss = self.alpha * loss_content \
                     + self.beta * loss_style + \
                     self.gamma * variation_loss

        return loss_content, loss_style, total_loss

    def train(self):
        img_generator, vgg = self.build_model()
        # Define the optimizer
        optimizer = optim.Adam(img_generator.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

        # Training loop
        for epoch in tqdm.tqdm(range(self.num_train_epochs)):
            img_generator.train()
            for step, batch in enumerate(self.dataloaders):
                content_imgs = batch['content_image']
                style_imgs = batch['style_image']

                # Generate stylized output
                stylized_output = img_generator(content_imgs)

                # Extract features from VGG19 for content and style images
                content_features = vgg(content_imgs)
                style_features = vgg(style_imgs)
                stylized_features = vgg(stylized_output)

                loss_content, loss_style, total_loss = self.compute_loss(content_features, style_features, stylized_features, stylized_output)

                # Backpropagation and optimization
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()

            # Print the loss for monitoring
            print(f"Epoch [{epoch + 1}/{self.num_train_epochs}], Total Loss: {total_loss.item()}")
            self.save(img_generator)

    def save(self, generator):
        model_path = os.path.join(self.output_dir, "model" + ".pth")
        torch.save(generator.state_dict(), model_path)



