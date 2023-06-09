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

import wandb
from PIL import Image
from typing import List
from tqdm.auto import tqdm

from src.models.losses import StyleLoss, TVLoss, HistLoss
from torch.nn import MSELoss as ContentLoss

from src.models.generator import Encoder, Decoder
from src.models.transformer import MTranspose
from src.utils.image_plot import plot_image


PRJ_NAME = "Style_transfer"


class Trainer:
    def __init__(self,
                 dataloaders,
                 output_dir: str,
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
                 delta: float = 2,
                 transformer_size: int = 32,
                 content_layers_idx: List[int] = [8, 11, 13],
                 style_layers_idx: List[int] = [1, 3, 6, 8]
                 ):

        self.output_dir = output_dir
        self.dataloaders = dataloaders.__call__()
        self.per_device_batch_size = per_device_batch_size

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.with_tracking = with_tracking
        self.login_key = login_key if with_tracking else None
        self.vgg_model_type = vgg_model_type
        self.learning_rate = learning_rate
        self.content_layers_idx = content_layers_idx
        self.style_layers_idx = style_layers_idx
        self.num_train_epochs = num_train_epochs
        self.transformer_size = transformer_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.variation_loss = TVLoss()
        self.style_loss = StyleLoss()
        self.content_loss = ContentLoss()
        self.hist_loss = HistLoss()

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
        encoder = Encoder().eval().to(self.device)
        decoder = Decoder().eval().to(self.device)

        transformer = MTranspose(matrix_size=self.transformer_size).to(self.device)

        content_extractors = self.get_feature_extractor(self.content_layers_idx, device=self.device)
        style_extractors = self.get_feature_extractor(self.style_layers_idx, device=self.device)

        return encoder, decoder, transformer, content_extractors, style_extractors

    def compute_loss(self, content_features, style_features, stylized_outputs, style_imgs):
        # Compute content loss
        losses = []
        for feature in content_features:
            loss = self.content_loss(feature['model_output'], feature['orginal'])
            losses.append(loss)
        loss_content = sum(losses)

        # Compute style loss
        losses = []
        for feature in style_features:
            loss = self.style_loss(feature['model_output'], feature['orginal'])
            losses.append(loss)
        loss_style_weightAdjust = map(lambda loss: loss*(1/(len(losses))), losses)
        loss_style = sum(loss_style_weightAdjust)

        # Compute variation loss
        variation_loss = self.variation_loss(stylized_outputs)

        # Compute hist loss
        histogram_loss = self.hist_loss(stylized_outputs, style_imgs)

        # Compute total loss
        total_loss = self.alpha * loss_content \
                     + self.beta * loss_style + \
                     self.gamma * variation_loss \
                     + self.delta * histogram_loss

        return loss_content, loss_style, variation_loss, histogram_loss, total_loss

    def train(self):
        encoder, decoder, transformer, content_extractors, style_extractors = self.build_model()

        # Define the optimizer
        transformer_optimizer  = optim.Adam(transformer.parameters(), lr=self.learning_rate)

        # Define the learning rate scheduler
        scheduler = lr_scheduler.LambdaLR(transformer_optimizer, lr_lambda=lambda epoch: 0.1 ** (epoch / 10))

        transformer.train()

        # Training loop
        loss_list = []
        for epoch in tqdm(range(self.num_train_epochs), desc="Training progress",
                          colour='green', position=0, leave=True, file=sys.stdout):
            for step, batch in enumerate(tqdm(self.dataloaders, colour='blue', desc="Training batch progress",
                                              position=1, leave=True, file=sys.stdout)):
                content_imgs = batch['content_image'].to(self.device)
                style_imgs = batch['style_image'].to(self.device)

                # Encode images
                encode_Cfeatures = encoder(content_imgs)
                encode_Sfeatures = encoder(style_imgs)

                transformed_features = transformer(encode_Cfeatures, encode_Sfeatures)

                # Decode features
                decode_imgs = decoder(transformed_features)

                # Extract features from VGG19 for content and style images
                content_features = []
                org_output_features = list(content_extractors(content_imgs).values())
                gen_output_features = list(content_extractors(decode_imgs).values())
                for layer_idx in range(len(self.content_layers_idx)):
                    content_features.append({"orginal": org_output_features[layer_idx],
                                             "model_output": gen_output_features[layer_idx]})

                style_features = []
                org_output_features = list(style_extractors(style_imgs).values())
                gen_output_features = list(style_extractors(decode_imgs).values())
                for layer_idx in range(len(self.style_layers_idx)):
                    style_features.append({"orginal": org_output_features[layer_idx],
                                             "model_output": gen_output_features[layer_idx]})

                loss_content, loss_style, variation_loss, histogram_loss, total_loss = self.compute_loss(content_features,
                                                                                        style_features,
                                                                                         decode_imgs,
                                                                                         style_imgs)

                del content_features, style_features, encode_Cfeatures, encode_Sfeatures, transformed_features

                # Backpropagation and optimization
                transformer_optimizer.zero_grad()
                total_loss.backward()
                transformer_optimizer.step()


            # Update learning rate
            scheduler.step()

            loss_list.append(float(total_loss.item()))
            # Print the loss for monitoring
            print(f"\n --- Training log --- \n")
            print(f"   Epoch [{epoch + 1}/{self.num_train_epochs}]"
                  f"\n Total Loss: {total_loss.item()}"
                  f"\n Loss content: {loss_content} | Contributed content loss: {loss_content*self.alpha}"
                  f"\n Loss style: {loss_style} | Contributed style loss: {loss_style*self.beta}"
                  f"\n Loss variation: {variation_loss} | Contributed variation loss: {variation_loss*self.gamma}"
                  f"\n Loss histogram: {histogram_loss} | Contributed histogram loss: {histogram_loss*self.delta}"
                  f"\n Minimum loss of overall training session: {min(loss_list)} \n")

            if self.with_tracking:
                print("--- Logging to wandb ---")
                self.wandb.log({"Loss_content_contributed": loss_content*self.alpha,
                                "Loss_style_contributed": loss_style*self.beta,
                                "Loss_variation_contributed": variation_loss*self.gamma,
                                "Loss_histogram_contributed": histogram_loss*self.delta,
                                "Total_loss": float(total_loss.item()),
                                "Min_loss": min(loss_list)}, step=epoch)

            if float(total_loss.item()) == min(loss_list):
                print(f"Saving epoch [{epoch + 1}/{self.num_train_epochs}]")
                self.save(transformer, loss_info={"total": float(total_loss.item()),
                                                    "content": loss_content,
                                                    "style": loss_style})
            else:
                print(f"Discarding epoch [{epoch + 1}/{self.num_train_epochs}]")
                continue

        if self.with_tracking:
            self.wandb.finish()

    def save(self, transformer, discriminator=None, loss_info=None):
        dir_name = f"TotalL_{loss_info['total']}_Content_{loss_info['content']}_Style_{loss_info['style']}"
        save_path_dir = os.path.join(self.output_dir, dir_name)
        print(f" Saving in {save_path_dir}")
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir, exist_ok=False)
        else:
            raise FileExistsError

        model_path = os.path.join(save_path_dir, f"transformer" + ".pth")
        torch.save(transformer.state_dict(), model_path)

        if discriminator is not None:
            model_path = os.path.join(self.output_dir, "discriminator" + ".pth")
            torch.save(discriminator.state_dict(), model_path)




