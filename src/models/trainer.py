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
from torchvision.models._utils import IntermediateLayerGetter

import tqdm
import wandb
from PIL import Image
from typing import List

from src.models.losses import StyleLoss, TVLoss
from torch.nn import MSELoss as ContentLoss

from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.utils.image_plot import plot_image


PRJ_NAME = "Style_transfer"


class Trainer:
    def __init__(self,
                 dataloaders,
                 output_dir: str,
                 lr_scheduler_type: str,
                 resume_from_checkpoint,
                 seed: int,
                 num_train_epochs: int,
                 weight_decay,
                 per_device_batch_size: int,
                 gradient_accumulation_steps: int,
                 do_eval_per_epoch: bool,
                 learning_rate: float,
                 alpha: float,
                 beta: float,
                 gamma: float,
                 login_key: str,
                 vgg_model_type: str = '19',
                 with_tracking: bool = False,
                 content_layers_idx: List[int] = [25, 30, 34],
                 style_layers_idx: List[int] = [14, 19, 25, 28, 34],
                 num_res_block: int = 20,
                 num_deep_res_block: int = 2
                 ):

        self.output_dir = output_dir
        self.dataloaders = dataloaders.__call__()
        self.per_device_batch_size = per_device_batch_size

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.with_tracking = with_tracking
        self.login_key = login_key if with_tracking else None
        self.vgg_model_type = vgg_model_type
        self.learning_rate = learning_rate
        self.num_res_block = num_res_block
        self.num_deep_res_block = num_deep_res_block
        self.content_layers_idx = content_layers_idx
        self.style_layers_idx = style_layers_idx
        self.num_train_epochs = num_train_epochs

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.variation_loss = TVLoss()
        self.style_loss = StyleLoss()
        self.content_loss = ContentLoss()

        if self.with_tracking:
            # Initialize tracker
            self.wandb.login(self.login_key)
            self.wandb = wandb.init(
                project=PRJ_NAME,
                # track hyperparameters and run metadata
                config={
                    "learning_rate": self.learning_rate,
                    "feature_extractor": self.vgg_model_type,
                    "loss_content_weight": self.alpha,
                    "loss_style_weight": self.beta,
                    "loss_variation_weight": self.gamma,
                    "epochs": self.num_train_epochs,
                    "content_layers_idx": self.content_layers_idx,
                    "style_layers_idx": self.style_layers_idx,
                    "num_deep_res_block": self.num_deep_res_block,
                    "num_res_block": self.num_res_block,
                    "device": self.device,
                    "batch_size": self.per_device_batch_size
                    }
                )
        else:
            self.wandb = None

    @staticmethod
    def get_feature_extractor(selected_indices, vgg_model_type="16", device='cpu'):
        if vgg_model_type == "16":
            vgg = models.vgg19(pretrained=True).features
        else:
            vgg = models.vgg16(pretrained=True).features

        layers = list(vgg.children())

        if isinstance(selected_indices, list):
            selected_layers = {str(idx): layers[idx] for idx in selected_indices if idx < len(layers)}
            if len(selected_layers) != len(selected_indices):
                raise ValueError("Invalid layer index provided.")
        else:
            raise ValueError("Selected indices should be provided as a list of layer indices.")

        feature_extractors = IntermediateLayerGetter(vgg, return_layers=selected_layers).to(device)
        return feature_extractors

    def build_model(self):
        generator = Generator(num_res_block=self.num_res_block,
                              num_deep_res_block=self.num_deep_res_block).to(self.device)
        discriminator = Discriminator().to(self.device)

        content_extractors = self.get_feature_extractor(self.content_layers_idx, device=self.device)
        style_extractors = self.get_feature_extractor(self.style_layers_idx, device=self.device)

        return generator, discriminator, content_extractors, style_extractors

    def compute_loss(self, content_features, style_features, stylized_output):
        # Compute content loss
        losses = []
        for feature in content_features:
            loss = self.content_loss(feature['orginal'], feature['model_output'])
            losses.append(loss)
        loss_content = sum(losses)

        # Compute style loss
        losses = []
        for feature in style_features:
            loss = self.style_loss(feature['orginal'], feature['model_output'])
            losses.append(loss)
        loss_style = sum(losses)

        # Compute variation loss
        variation_loss = self.variation_loss(stylized_output)

        # Compute total loss
        total_loss = self.alpha * loss_content \
                     + self.beta * loss_style + \
                     self.gamma * variation_loss

        return loss_content, loss_style, variation_loss, total_loss

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
                # some context of what to style the content on (batch, channels, width, height)
                stacked_images = torch.cat((content_imgs, style_imgs), dim=1)

                # Generate stylized output
                stylized_output = img_generator(stacked_images)

                # Extract features from VGG19 for content and style images
                content_features = []
                org_output_features = list(content_extractors(content_imgs).values())
                gen_output_features = list(content_extractors(stylized_output).values())
                for layer_idx in range(len(self.content_layers_idx)):
                    content_features.append({"orginal": org_output_features[layer_idx],
                                             "model_output": gen_output_features[layer_idx]})

                style_features = []
                org_output_features = list(style_extractors(style_imgs).values())
                gen_output_features = list(style_extractors(stylized_output).values())
                for layer_idx in range(len(self.style_layers_idx)):
                    style_features.append({"orginal": org_output_features[layer_idx],
                                             "model_output": gen_output_features[layer_idx]})

                loss_content, loss_style, variation_loss, total_loss = self.compute_loss(content_features,
                                                                                        style_features, stylized_output)

                # Backpropagation and optimization
                generator_optimizer.zero_grad()
                total_loss.backward()
                generator_optimizer.step()
                scheduler.step()

            loss_list.append(float(total_loss.item()))
            # Print the loss for monitoring
            print(f"\n --- Training log --- \n")
            print(f"   Epoch [{epoch + 1}/{self.num_train_epochs}]"
                  f"\n Total Loss: {total_loss.item()}"
                  f"\n Loss content: {loss_content} | Contributed content loss: {loss_content*self.alpha}"
                  f"\n Loss style: {loss_style} | Contributed style loss: {loss_style*self.beta}"
                  f"\n Loss variation: {variation_loss} | Contributed variation loss: {variation_loss*self.gamma}"
                  f"\n Minimum loss of overall training session: {min(loss_list)} \n")

            if self.with_tracking:
                print("--- Logging to wandb ---")
                self.wandb.log({"Loss_content_contributed": loss_content*self.alpha,
                                "Loss_style_contributed": loss_style*self.beta,
                                "Loss_variation_contributed": variation_loss*self.gamma,
                                "Total_loss": float(total_loss.item()),
                                "Min_loss": min(loss_list)}, step=epoch)

            if float(total_loss.item()) == min(loss_list):
                print(f"Saving epoch [{epoch + 1}/{self.num_train_epochs}]")
                self.save(img_generator, loss_info={"total": float(total_loss.item()),
                                                    "content": loss_content,
                                                    "style": loss_style})
            else:
                print(f"Discarding epoch [{epoch + 1}/{self.num_train_epochs}]")
                continue

        if self.with_tracking:
            self.wandb.finish()

    def save(self, generator, discriminator=None, loss_info=None):
        dir_name = f"TotalL_{loss_info['total']}_Content_{loss_info['content']}_Style_{loss_info['style']}"
        save_path_dir = os.path.join(self.output_dir, dir_name)
        print(f" Saving in {save_path_dir}")
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir, exist_ok=False)
        else:
            raise FileExistsError

        model_path = os.path.join(save_path_dir, f"generator" + ".pth")
        torch.save(generator.state_dict(), model_path)

        if discriminator is not None:
            model_path = os.path.join(self.output_dir, "discriminator" + ".pth")
            torch.save(discriminator.state_dict(), model_path)



