import io
import random
import sys
import os
import time
import math
import warnings

sys.path.insert(0,r'./') #Add root directory here

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models._utils import IntermediateLayerGetter

import wandb
import numpy as np
from PIL import Image
from typing import List
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from src.models.losses import StyleLoss, TVLoss, HistLoss
from torch.nn import MSELoss as ContentLoss

from src.models.generator import Encoder, Decoder
from src.models.transformer import MTranspose
from src.utils.image_plot import plot_image
from src.utils.utils import set_seed


PRJ_NAME = "Style_transfer"


class Trainer:
    def __init__(self,
                 dataloaders,
                 output_dir: str,
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
                 seed: int = 42,
                 warm_up_epoch: int = 1,
                 plot_per_epoch: bool = False,
                 save_best: bool = True,
                 resume_from_checkpoint: str = None,
                 vgg_model_type: str = '19',
                 with_tracking: bool = False,
                 log_weights_cpkt: bool = False,
                 delta: float = 2,
                 step_frequency: float = 0.5,
                 transformer_size: int = 32,
                 layer_depth: int = 1,
                 deep_learner: bool = False,
                 content_layers_idx: List[int] = [11, 17, 22, 26],
                 style_layers_idx: List[int] = [1, 3, 6, 8, 9, 11]
                 ):

        self.output_dir = output_dir
        self.dataloaders = dataloaders.__call__()
        self.per_device_batch_size = per_device_batch_size

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.with_tracking = with_tracking
        self.log_weights_cpkt = log_weights_cpkt
        self.plot_per_epoch = plot_per_epoch
        self.save_best = save_best
        self.login_key = login_key if with_tracking else None
        self.vgg_model_type = vgg_model_type
        self.resume_from_checkpoint = resume_from_checkpoint
        self.learning_rate = learning_rate
        self.content_layers_idx = content_layers_idx
        self.style_layers_idx = style_layers_idx
        self.num_train_epochs = num_train_epochs
        self.transformer_size = transformer_size
        self.layer_depth = layer_depth
        self.deep_learner = deep_learner
        self.warm_up_epoch = warm_up_epoch
        self.step_frequency = step_frequency

        set_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.variation_loss = TVLoss()
        self.style_loss = StyleLoss()
        self.content_loss = ContentLoss()
        self.hist_loss = HistLoss()

        if self.with_tracking:
            if wandb.__version__ != '0.13.9':
                raise "Wandb mismatch error!, please install the correct version as the artifact log of " \
                      "this implementation only support for this particular version"
            try:
                print(f"--- Initializing wandb {PRJ_NAME} --- ")
                # Initialize tracker
                self.wandb = wandb.init(
                    project=PRJ_NAME,
                    # track hyperparameters and run metadata
                    config={
                        "learning_rate": self.learning_rate,
                        "feature_extractor": self.vgg_model_type,
                        "loss_content_weight": self.alpha,
                        "loss_style_weight": self.beta,
                        "loss_variation_weight": self.gamma,
                        "loss_histogram_weight": self.delta,
                        "epochs": self.num_train_epochs,
                        "content_layers_idx": self.content_layers_idx,
                        "style_layers_idx": self.style_layers_idx,
                        "device": self.device,
                        "batch_size": self.per_device_batch_size,
                        "layer_depth": self.layer_depth,
                        "deep_learner": self.deep_learner,
                        "transformer_size": self.transformer_size,
                        "step_frequency": self.step_frequency,
                        "num_batch": len(self.dataloaders),
                        "num_exampes": len(self.dataloaders.dataset)
                        }
                )
                if self.log_weights_cpkt:
                    try:
                        self.artifact = wandb.Artifact(name=f"Checkpoints_{PRJ_NAME}",
                                                   type='model', description="Model checkpoint for style transfer")
                        print(f"\n Creating new artifact to save cpkt...")
                    except Exception:
                        try:
                            self.artifact = self.wandb.use_artifact(artifact_or_name=f"Checkpoints_{PRJ_NAME}")
                            print(f"\n Using saved artifact to save cpkt...")
                        except Exception:
                            raise "Unable to creating or initializing artifact to log cpkt!"
                else:
                    self.artifact = None

            except Exception:
                raise "Not login yet!"
        else:
            self.wandb = None

    @staticmethod
    def get_feature_extractor(selected_indices, vgg_model_type="16", device='cpu'):
        if vgg_model_type == "16":
            vgg = models.vgg19(weights='IMAGENET1K_V1').features
        else:
            vgg = models.vgg16(weights='IMAGENET1K_V1').features

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

        transformer = MTranspose(matrix_size=self.transformer_size,
                                 layer_depth=self.layer_depth, deep_learner=self.deep_learner).to(self.device)

        content_extractors = self.get_feature_extractor(self.content_layers_idx, device=self.device)
        style_extractors = self.get_feature_extractor(self.style_layers_idx, device=self.device)

        return encoder, decoder, transformer, content_extractors, style_extractors

    def compute_loss(self, content_features, style_features, stylized_outputs, style_imgs):
        # Compute content loss
        losses = []
        for feature in content_features:
            loss = self.content_loss(feature['model_output'], feature['orginal'])
            losses.append(loss)
        loss_content_weightAdjust = map(lambda loss: loss*(1/(len(losses))), losses)
        loss_content = sum(loss_content_weightAdjust)

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
        transformer_optimizer = optim.Adam(transformer.parameters(), lr=self.learning_rate)

        # Define the learning rate scheduler
        def lr_lambda(epoch, warm_up_epoch):
            if epoch < warm_up_epoch:
                return epoch / warm_up_epoch
            else:
                return 0.1 ** ((epoch - warm_up_epoch) / 10)

        scheduler = lr_scheduler.LambdaLR(transformer_optimizer,
                                          lr_lambda=lambda epoch: lr_lambda(epoch, self.warm_up_epoch))

        init_epoch = 0
        loss_list = []
        completed_step = 0
        if self.resume_from_checkpoint is not None:
            checkpoint_dir = os.path.basename(os.path.dirname(os.path.realpath(self.resume_from_checkpoint)))
            print(f"\n --- Resume from checkpoint {checkpoint_dir}--- \n")
            try:
                checkpoint = torch.load(self.resume_from_checkpoint, map_location=self.device)
            except Exception:
                raise "Unable to load from checkpoint, invalid path!"

            print(f"\n Loading transformer model... ")
            try:
                transformer.load_state_dict(checkpoint['model_state_dict'])
            except Exception:
                raise "Unable to load weight to the model, wrong model structure compare to checkpoint!"

            print(f"\n Loading optimizer...")
            try:
                transformer_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                raise "Unable to load optimizer state!"

            try:
                criterion = checkpoint['criterion']
                total_loss = checkpoint['loss']
                print(f"\n Loss from previous training session: {total_loss}")
                if criterion['alpha'] == self.alpha and criterion['beta'] == self.beta\
                        and criterion['gamma'] == self.gamma and criterion['delta'] == self.delta\
                        and criterion['content_layers_idx'] == self.content_layers_idx \
                        and criterion['style_layers_idx'] == self.style_layers_idx:
                    loss_list.append(total_loss)
                else:
                    warnings.warn("Last session loss discarded due to changed loss criteria")
            except KeyError:
                warnings.warn("checkpoint missing criterion info!")
                pass

            try:
                last_session_completed_step = checkpoint['completed_step']
                completed_step += last_session_completed_step

                last_session_epoch = checkpoint['epoch']
                print(f"\n Last training session epoch: {last_session_epoch + 1}")

                if last_session_epoch + 1 < self.num_train_epochs:
                    init_epoch = last_session_epoch + 1
                else:
                    raise "Num train epoch can't be smaller than last session epoch resume from checkpoint!"
            except KeyError:
                warnings.warn("Checkpoint missing train process info!")
                pass

        # Calculate how often should we update the lr
        total_steps = len(self.dataloaders) * (self.num_train_epochs-init_epoch)
        steps_per_epoch = len(self.dataloaders)
        scheduler_steps = math.ceil(steps_per_epoch *  self.step_frequency)
        scheduler_count = 0

        print(f"\n --- Training init log --- \n")
        print(f"\n Number of epoch: {self.num_train_epochs}"
              f"\n Init epoch: {init_epoch}"
              f"\n Batch size: {self.per_device_batch_size}"
              f"\n Total number of batch: {len(self.dataloaders)}"
              f"\n Total number of examples: {len(self.dataloaders.dataset)}"
              f"\n Total number of steps: {total_steps}"
              f"\n Number of steps before updating the lr: {scheduler_steps}"
              f"\n Transformation matrix size: {self.transformer_size}"
              f"\n Content layers loss idx: {self.content_layers_idx}"
              f"\n Style layers loss idx: {self.style_layers_idx}"
              f"\n Depth of CNN layer: {self.layer_depth}"
              f"\n Deep learner: {self.deep_learner}"
              f"\n Device to train: {self.device}\n")

        # Training loop
        for epoch in tqdm(range(init_epoch, self.num_train_epochs), desc="Training progress",
                          colour='green', position=0, leave=True, file=sys.stdout):
            total_epoch_loss, total_epoch_content_loss = 0, 0
            total_epoch_style_loss, total_epoch_var_loss = 0, 0
            total_epoch_hist_loss = 0
            transformer.train()
            for step, batch in enumerate(tqdm(self.dataloaders, colour='blue', desc="Training batch progress",
                                              position=1, leave=False, file=sys.stdout)):
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

                del content_features, style_features, encode_Cfeatures, encode_Sfeatures
                del org_output_features, gen_output_features, transformed_features

                # Backpropagation and optimization
                transformer_optimizer.zero_grad()
                total_loss.backward()
                transformer_optimizer.step()

                completed_step += step
                total_epoch_loss += float(total_loss.item())
                total_epoch_content_loss += float(loss_content.item())
                total_epoch_style_loss += float(loss_style.item())
                total_epoch_var_loss += float(variation_loss.item())
                total_epoch_hist_loss += float(histogram_loss.item())

                if self.with_tracking:
                    self.wandb.log({"Loss_content_contributed_batch": loss_content * self.alpha,
                                    "Loss_style_contributed_batch": loss_style * self.beta,
                                    "Loss_variation_contributed_batch": variation_loss * self.gamma,
                                    "Loss_histogram_contributed_batch": histogram_loss * self.delta,
                                    "Total_loss_batch": float(total_loss.item())}, step=completed_step)

                # Update learning rate
                scheduler_count += 1
                if scheduler_count >= scheduler_steps:
                    print(f"\n --- Learning rate update --- \n")
                    scheduler.step()
                    scheduler_count = 0

            avg_epoch_total_loss = total_epoch_loss / len(self.dataloaders)
            avg_epoch_content_loss = total_epoch_content_loss / len(self.dataloaders)
            avg_epoch_style_loss = total_epoch_style_loss / len(self.dataloaders)
            avg_epoch_var_loss = total_epoch_var_loss / len(self.dataloaders)
            avg_epoch_hist_loss = total_epoch_hist_loss / len(self.dataloaders)

            loss_list.append(avg_epoch_total_loss)
            # Print the loss for monitoring
            print(f"\n --- Training log --- \n")
            print(f"   Epoch [{epoch + 1}/{self.num_train_epochs}]"
                  f"\n Total Loss: {avg_epoch_total_loss}"
                  f"\n Loss content: {avg_epoch_content_loss} | Contributed content loss: {avg_epoch_content_loss*self.alpha}"
                  f"\n Loss style: {avg_epoch_style_loss} | Contributed style loss: {avg_epoch_style_loss*self.beta}"
                  f"\n Loss variation: {avg_epoch_var_loss} | Contributed variation loss: {avg_epoch_var_loss*self.gamma}"
                  f"\n Loss histogram: {avg_epoch_hist_loss} | Contributed histogram loss: {avg_epoch_hist_loss*self.delta}"
                  f"\n Minimum loss of overall training session: {min(loss_list)} \n")

            if self.plot_per_epoch:
                print(f"\n Plotting comparison of epoch [{epoch + 1}/{self.num_train_epochs}]... \n")
            plot = self.plot_comparison(encoder, decoder, transformer,
                                         r"./src/data/dummy/content/im1.jpg", r"./src/data/dummy/style/85343.jpg",
                                         transforms.Compose([
                                             transforms.Resize(256),
                                             transforms.ToTensor()
                                         ]), self.device, plot=self.plot_per_epoch)

            if self.with_tracking:
                print("--- Logging to wandb ---")
                self.wandb.log({"Epoch": epoch,
                                "Loss_content_contributed": avg_epoch_content_loss*self.alpha,
                                "Loss_style_contributed": avg_epoch_style_loss*self.beta,
                                "Loss_variation_contributed": avg_epoch_var_loss*self.gamma,
                                "Loss_histogram_contributed": avg_epoch_hist_loss*self.delta,
                                "Total_loss": avg_epoch_total_loss,
                                "Min_loss": min(loss_list)}, step=completed_step)
                try:
                    io_buf = io.BytesIO()
                    plot.savefig(io_buf, format='raw')
                    io_buf.seek(0)
                    comparison_plot = wandb.Image(Image.frombytes('RGBA', (1000, 400), io_buf.getvalue(), 'raw'),
                                                  caption=f"Comparison of epoch {epoch}")
                    io_buf.close()
                    self.wandb.log({"Examples": comparison_plot}, step=completed_step)
                except Exception:
                    warnings.warn("Unable to log examples to wandb")
                    pass

            if avg_epoch_total_loss == min(loss_list) and self.save_best:
                print(f"Saving epoch [{epoch + 1}/{self.num_train_epochs}]")
                self.save(transformer, transformer_optimizer, info={"total": avg_epoch_total_loss,
                                                                    "content": avg_epoch_content_loss,
                                                                    "style": avg_epoch_style_loss,
                                                                    "epoch": epoch,
                                                                    "completed_step": completed_step
                                                                    }, result=plot)
            elif not self.save_best:
                print(f"Saving epoch [{epoch + 1}/{self.num_train_epochs}]")
                self.save(transformer, transformer_optimizer, info={"total": avg_epoch_total_loss,
                                                                    "content": avg_epoch_content_loss,
                                                                    "style": avg_epoch_style_loss,
                                                                    "epoch": epoch,
                                                                    "completed_step": completed_step
                                                                    }, result=plot)
            else:
                print(f"Discarding epoch [{epoch + 1}/{self.num_train_epochs}]")
                plot.close('all')
                continue

        if self.with_tracking:
            self.wandb.finish()

    def save(self, transformer, transformer_optimizer,
                   discriminator=None, info=None, result=None):
        dir_name = f"TotalL_{info['total']}_Content_{info['content']}_Style_{info['style']}"
        save_path_dir = os.path.join(self.output_dir, dir_name)
        print(f" Saving in {save_path_dir}")
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir, exist_ok=False)
        else:
            warnings.warn("File exist!, might be a good idea to change seed")
            raise FileExistsError

        model_path = os.path.join(save_path_dir, f"transformer{len(self.dataloaders.dataset)}" + ".pth")
        plot_path = os.path.join(save_path_dir, "result_plot.png")
        # Save weights at output dir
        torch.save({
            'epoch': info['epoch'],
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': transformer_optimizer.state_dict(),
            'loss': info['total'],
            'completed_step': info['completed_step'],
            "trans_size": self.transformer_size,
            "layer_depth": self.layer_depth,
            "deep_learner": self.deep_learner,
            "criterion": {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "delta": self.delta,
                "content_layers_idx": self.content_layers_idx,
                "style_layers_idx": self.style_layers_idx
            }
        }, model_path)
        result.savefig(plot_path)

        if self.with_tracking and self.log_weights_cpkt:
            print(f" --- Saving {PRJ_NAME} checkpoint to wandb ---")
            try:
                self.artifact.add_dir(local_path=save_path_dir, name=f"Checkpoint_{PRJ_NAME}_{dir_name}")
                self.wandb.log_artifact(self.artifact)
            except ValueError:
                try:
                    self.artifact = wandb.Artifact(name=f"Checkpoints_{PRJ_NAME}",
                                                   type='model', description="Model checkpoint for style transfer")
                    print(f"\n Using saved artifact to save cpkt...")
                    self.artifact.add_dir(local_path=save_path_dir, name=f"Checkpoint_{PRJ_NAME}_{dir_name}")
                    self.wandb.log_artifact(self.artifact)
                except Exception:
                    raise "Unable to creating or initializing artifact to log cpkt!"

        if discriminator is not None:
            model_path = os.path.join(self.output_dir, "discriminator" + ".pth")
            torch.save(discriminator.state_dict(), model_path)

    @staticmethod
    def plot_comparison(encoder, decoder, transformer, content_img_url, style_img_url,
                        transformation, device, sleep: int = 5, plot: bool = True):
        try:
            content_image = Image.open(content_img_url).convert('RGB')
            style_image = Image.open(style_img_url).convert('RGB')
        except IOError:
            raise "Invalid image url!"

        content_image_tensor = transformation(content_image).to(device)
        style_image_tensor = transformation(style_image).to(device)

        transformer.eval()
        decoder.eval()

        encode_Cfeatures = encoder(content_image_tensor.unsqueeze(0))
        encode_Sfeatures = encoder(style_image_tensor.unsqueeze(0))

        transformed_features = transformer(encode_Cfeatures, encode_Sfeatures)

        decode_img = decoder(transformed_features)

        # Convert tensor images to numpy arrays and adjust their shape if needed
        image1_np = content_image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()  # Adjust dimensions as per your tensor shape
        image2_np = style_image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        image3_np = decode_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()  # Adjust dimensions as per your tensor shape

        # Create a figure with subplots
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))

        # Normalize 0 - 1
        image1_np = np.interp(image1_np, (image1_np.min(), image1_np.max()), (0, 1))
        image2_np = np.interp(image2_np, (image2_np.min(), image2_np.max()), (0, 1))
        image3_np = np.interp(image3_np, (image3_np.min(), image3_np.max()), (0, 1))

        # Plot each image on a separate subplot
        axes[0].imshow(image1_np)
        axes[0].set_title('Content')
        axes[0].axis('off')

        axes[1].imshow(image2_np)
        axes[1].set_title('Style')
        axes[1].axis('off')

        axes[2].imshow(image3_np)
        axes[2].set_title('Stylized content')
        axes[2].axis('off')

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the figure (optional)
        if plot:
            plt.show(block=False)
            plt.pause(sleep)

        return plt
