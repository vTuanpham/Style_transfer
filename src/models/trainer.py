import io
import random
import datetime
import sys
import os
import time
import math
import warnings
from argparse import Namespace
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

from src.models.losses import AdaINStyleLoss, AdaINContentLoss, TVLoss, HistLoss
from src.models.generator import Encoder, Decoder
from src.models.transformer import AdaIN
from src.utils.custom_transform import RGBToGrayscaleStacked



PRJ_NAME = "Style_transfer"


class Trainer:
    def __init__(self,
                 dataloaders,
                 output_dir: str,
                 num_train_epochs: int,
                 per_device_batch_size: int,
                 learning_rate: float,
                 alpha: float,
                 beta: float,
                 gamma: float,
                 seed: int = 42,
                 do_eval_per_epoch: bool = True,
                 plot_per_epoch: bool = False,
                 save_best: bool = True,
                 resume_from_checkpoint: str = None,
                 vgg_model_type: str = '19',
                 optim_name: dict = {'optim_name': 'adam'},
                 with_tracking: bool = False,
                 log_weights_cpkt: bool = False,
                 do_decoder_train: bool = False,
                 use_pretrained_WCTDECODER: bool = False,
                 delta: float = 2,
                 eps: float = 1e-5,
                 step_frequency: float = 0.5,
                 gradient_threshold: float = None,
                 config: Namespace = None,
                 grayscale_content_transform: bool=False,
                 content_layers_idx: List[int] = [12, 16, 21],
                 style_layers_idx: List[int] = [0, 5, 10, 19, 28]
                 ):

        self.output_dir = output_dir
        self.dataloaders = dataloaders.__call__()
        self.per_device_batch_size = per_device_batch_size

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eps = eps

        self.vgg_model_type = vgg_model_type
        self.learning_rate = learning_rate
        self.content_layers_idx = content_layers_idx
        self.style_layers_idx = style_layers_idx
        self.num_train_epochs = num_train_epochs
        self.step_frequency = step_frequency
        self.optim_name = optim_name
        self.gradient_threshold = gradient_threshold
        self.do_decoder_train = do_decoder_train
        self.use_pretrained_WCTDECODER = use_pretrained_WCTDECODER
        self.grayscale_content_transform = grayscale_content_transform

        self.with_tracking = with_tracking
        self.log_weights_cpkt = log_weights_cpkt
        self.plot_per_epoch = plot_per_epoch
        self.save_best = save_best
        self.resume_from_checkpoint = resume_from_checkpoint
        self.do_eval_per_epoch = do_eval_per_epoch
        self.config = config

        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.variation_loss = TVLoss()
        self.style_loss = AdaINStyleLoss()
        self.content_loss = AdaINContentLoss()
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
                        "step_frequency": self.step_frequency,
                        "num_batch": len(self.dataloaders['train']),
                        "num_exampes": len(self.dataloaders['train'].dataset),
                        "gradient_threshold": self.gradient_threshold,
                        "optim_name": self.optim_name,
                        "step_frequency": self.step_frequency,
                        "seed": self.seed,
                        "do_decoder_train": self.do_decoder_train,
                        "use_pretrained_WCTDECODER": self.use_pretrained_WCTDECODER,
                        "grayscale_content_transform": self.grayscale_content_transform,
                        "full_config": vars(self.config)
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

    @staticmethod
    def get_optimizer(optimizer_name: str, parameters, learning_rate: float=5e-5, kwargs: dict=None):
        optimizer_name = optimizer_name.lower()

        kwargs_clone = kwargs.copy()
        del kwargs_clone['optim_name']

        try:
            if optimizer_name == 'sgd':
                return optim.SGD(parameters, lr=learning_rate, **kwargs_clone)
            elif optimizer_name == 'adam':
                return optim.Adam(parameters, lr=learning_rate, **kwargs_clone)
            elif optimizer_name == 'rmsprop':
                return optim.RMSprop(parameters, lr=learning_rate, **kwargs_clone)
            elif optimizer_name == 'adagrad':
                return optim.Adagrad(parameters, lr=learning_rate, **kwargs_clone)
            elif optimizer_name == 'adadelta':
                return optim.Adadelta(parameters, lr=learning_rate, **kwargs_clone)
            elif optimizer_name == 'adamw':
                return optim.AdamW(parameters, lr=learning_rate, **kwargs_clone)
            elif optimizer_name == 'adamax':
                return optim.Adamax(parameters, lr=learning_rate, **kwargs_clone)
            elif optimizer_name == 'sparseadam':
                return optim.SparseAdam(parameters, lr=learning_rate, **kwargs_clone)
            elif optimizer_name == 'lbfgs':
                return optim.LBFGS(parameters, lr=learning_rate, **kwargs_clone)
            elif optimizer_name == 'rprop':
                return optim.Rprop(parameters, lr=learning_rate, **kwargs_clone)
            else:
                raise ValueError(f"Optimizer '{optimizer_name}' not supported.")
        except TypeError:
            raise "Invalid argument for optimizer!"

    def build_model(self):
        encoder = Encoder().eval().to(self.device)
        decoder = Decoder(use_pretrained_WCT=self.use_pretrained_WCTDECODER if self.resume_from_checkpoint is None else False,
                          do_train=self.do_decoder_train).to(self.device)

        transformer = AdaIN(eps=self.eps).to(self.device)

        content_extractors = self.get_feature_extractor(self.content_layers_idx, device=self.device)
        style_extractors = self.get_feature_extractor(self.style_layers_idx, device=self.device)

        return encoder, decoder, transformer, content_extractors, style_extractors

    def compute_loss(self, content_features, style_features, stylized_outputs, style_imgs):
        # Compute content loss
        losses = []
        for feature in content_features:
            loss = self.content_loss(feature['model_output'], feature['orginal'])
            losses.append(loss)
        # loss_content_weightAdjust = map(lambda loss: loss*(1/(len(losses))), losses)
        # loss_content = sum(loss_content_weightAdjust)
        loss_content = sum(losses)

        # Compute style loss
        losses = []
        for feature in style_features:
            loss = self.style_loss(feature['model_output'], feature['orginal'])
            losses.append(loss)
        # loss_style_weightAdjust = map(lambda loss: loss*(1/(len(losses))), losses)
        # loss_style = sum(loss_style_weightAdjust)
        loss_style = sum(losses)

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
        if self.do_decoder_train:
            transformer_optimizer = self.get_optimizer(optimizer_name=self.optim_name['optim_name'],
                                                       parameters=list(transformer.parameters()) + list(decoder.parameters()),
                                                       learning_rate=self.learning_rate,
                                                       kwargs=self.optim_name
                                                       )
        else:
            transformer_optimizer = self.get_optimizer(optimizer_name=self.optim_name['optim_name'],
                                                       parameters=transformer.parameters(),
                                                       learning_rate=self.learning_rate,
                                                       kwargs=self.optim_name
                                                       )

        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

            try:
                if checkpoint['decoder_state_dict'] is not None:
                    print(f"\n Loading decoder model...")
                    decoder.load_state_dict(checkpoint['decoder_state_dict'])
            except Exception:
                raise "Unable to load weight to the decoder model, wrong model structure compare to checkpoint!"

            print(f"\n Loading optimizer...")
            try:
                if self.optim_name['optim_name'] == checkpoint['optim_name']['optim_name']:
                    transformer_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    if not self.do_decoder_train:
                        transformer_optimizer = self.get_optimizer(optimizer_name=checkpoint['optim_name']['optim_name'],
                                                                   parameters=transformer.parameters(),
                                                                   learning_rate=self.learning_rate,
                                                                   kwargs=checkpoint['optim_name']
                                                                   )
                        transformer_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    else:
                        transformer_optimizer = self.get_optimizer(optimizer_name=checkpoint['optim_name']['optim_name'],
                                                                   parameters=list(transformer.parameters()) + list(decoder.parameters()),
                                                                   learning_rate=self.learning_rate,
                                                                   kwargs=checkpoint['optim_name']
                                                                   )
                        transformer_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    warnings.warn(f"Set optim {self.optim_name['optim_name']} different from checkpoint optim {checkpoint['optim_name']['optim_name']},"
                                  f" switching to {checkpoint['optim_name']['optim_name']}")
                    self.optim_name['optim_name'] = checkpoint['optim_name']['optim_name']
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

        scheduler = lr_scheduler.CosineAnnealingLR(transformer_optimizer,
                                            T_max=len(self.dataloaders['train']) * (self.num_train_epochs-init_epoch),
                                            last_epoch=last_session_epoch if self.resume_from_checkpoint else -1)

        # Calculate how often should we update the lr
        total_steps = len(self.dataloaders['train']) * (self.num_train_epochs-init_epoch)
        steps_per_epoch = len(self.dataloaders['train'])
        scheduler_steps = math.ceil(steps_per_epoch * self.step_frequency)
        scheduler_count = 0

        # Register backward hook for gradient clipping, this prevents the gradient from exploding
        # (recommended range 1-10)
        if self.gradient_threshold is not None:
            for p in transformer.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -self.gradient_threshold, self.gradient_threshold))

        print(f"\n --- Training init log --- \n")
        print(f"\n Number of epoch: {self.num_train_epochs}"
              f"\n Optim name: {self.optim_name}"
              f"\n Gradient threshold: {self.gradient_threshold}"
              f"\n Grayscale convert for content: {self.grayscale_content_transform}"
              f"\n Init epoch: {init_epoch}"
              f"\n Eps: {self.eps}"
              f"\n Do decoder training: {self.do_decoder_train}"
              f"\n Load pretrained WCT image recover: {self.use_pretrained_WCTDECODER}"
              f"\n Batch size: {self.per_device_batch_size}"
              f"\n Total number of batch: {len(self.dataloaders['train'])}"
              f"\n Total number of examples: {len(self.dataloaders['train'].dataset)}"
              f"\n Total number of steps: {total_steps}"
              f"\n Number of steps before updating the lr: {scheduler_steps}"
              f"\n Content layers loss idx: {self.content_layers_idx}"
              f"\n Style layers loss idx: {self.style_layers_idx}"
              f"\n Device to train: {self.device}\n")

        # Training loop
        progress_bar = tqdm(range(init_epoch, self.num_train_epochs), desc="Training progress",
                            colour='green', position=0, leave=True)
        for epoch in progress_bar:
            total_epoch_loss, total_epoch_content_loss = 0, 0
            total_epoch_style_loss, total_epoch_var_loss = 0, 0
            total_epoch_hist_loss = 0
            transformer.train()
            if self.do_decoder_train: decoder.train()
            for step, batch in enumerate(tqdm(self.dataloaders['train'], colour='blue',
                                              desc="Training batch progress", position=1, leave=False)):
                content_imgs = batch['content_image'].to(self.device)
                style_imgs = batch['style_image'].to(self.device)

                # Encode images
                encode_Cfeatures = encoder(content_imgs)
                encode_Sfeatures = encoder(style_imgs)

                transformed_features = transformer(encode_Cfeatures, encode_Sfeatures)
                transformed_features = 1.0 * transformed_features + (1 - 1.0) * encode_Cfeatures

                # Decode features
                decode_imgs = decoder(transformed_features)

                # Normalize output
                content_imgs = norm(content_imgs)
                style_imgs = norm(style_imgs)
                decode_imgs = norm(decode_imgs)

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

                loss_content, loss_style, variation_loss, histogram_loss, total_loss = self.compute_loss(
                                                                                        content_features,
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

                epss = transformer.get_current_eps()

                if self.with_tracking:
                    rate = progress_bar.format_dict["rate"]
                    remaining = (progress_bar.total - progress_bar.n) / rate if rate and progress_bar.total else 0
                    self.wandb.log({"Elapsed(hours)": progress_bar.format_dict['elapsed'] / 60 / 60,
                                    "Time_left(hours)": remaining / 60 / 60,
                                    "Loss_content_contributed_batch": loss_content * self.alpha,
                                    "Loss_style_contributed_batch": loss_style * self.beta,
                                    "Loss_variation_contributed_batch": variation_loss * self.gamma,
                                    "Loss_histogram_contributed_batch": histogram_loss * self.delta,
                                    "Total_loss_batch": float(total_loss.item()),
                                    "Style_EPS": epss[0],
                                    "Content_EPS": epss[1]}, step=completed_step)

                # Update learning rate
                scheduler_count += 1
                if scheduler_count >= scheduler_steps:
                    print(f"\n --- Learning rate update --- \n")
                    scheduler.step()
                    scheduler_count = 0

            avg_epoch_total_loss = total_epoch_loss / len(self.dataloaders['train'])
            avg_epoch_content_loss = total_epoch_content_loss / len(self.dataloaders['train'])
            avg_epoch_style_loss = total_epoch_style_loss / len(self.dataloaders['train'])
            avg_epoch_var_loss = total_epoch_var_loss / len(self.dataloaders['train'])
            avg_epoch_hist_loss = total_epoch_hist_loss / len(self.dataloaders['train'])

            loss_list.append(avg_epoch_total_loss)

            epss = transformer.get_current_eps()
            # Print the loss for monitoring
            print(f"\n --- Training log --- \n")
            print(f"   Epoch [{epoch + 1}/{self.num_train_epochs}]"
                  f"\n EPS: Style: {epss[0]} "
                  f"\n      Actual eps style weight: {transformer.style_eps.item()}"
                  f"\n      Content: {epss[1]} "
                  f"\n      Actual eps content weight:{transformer.content_eps.item()}"
                  f"\n Total Loss: {avg_epoch_total_loss}"
                  f"\n Loss content: {avg_epoch_content_loss} | Contributed content loss: {avg_epoch_content_loss*self.alpha}"
                  f"\n Loss style: {avg_epoch_style_loss} | Contributed style loss: {avg_epoch_style_loss*self.beta}"
                  f"\n Loss variation: {avg_epoch_var_loss} | Contributed variation loss: {avg_epoch_var_loss*self.gamma}"
                  f"\n Loss histogram: {avg_epoch_hist_loss} | Contributed histogram loss: {avg_epoch_hist_loss*self.delta}"
                  f"\n Minimum loss of overall training session: {min(loss_list)} \n")

            if self.plot_per_epoch:
                print(f"\n Plotting comparison of epoch [{epoch + 1}/{self.num_train_epochs}]... \n")
            plots = None
            if self.do_eval_per_epoch and avg_epoch_total_loss == min(loss_list) and self.save_best:
                plots = []
                for step, batch in enumerate(tqdm(self.dataloaders['eval'], colour='red',
                                                  desc="Evaluating progress", position=2, leave=False)):
                    content_img_paths = batch['content_image']
                    style_img_paths = batch['style_image']

                    plot, _ = self.plot_comparison(encoder, decoder, transformer,
                                                 content_img_paths, style_img_paths,
                                                 content_transformation=transforms.Compose([
                                                     transforms.Resize(256),
                                                     transforms.ToTensor(),
                                                     RGBToGrayscaleStacked(enable=self.grayscale_content_transform)
                                                 ]),
                                                 style_transformation=transforms.Compose([
                                                     transforms.Resize(256),
                                                     transforms.ToTensor()
                                                 ]),
                                                device=self.device, plot=self.plot_per_epoch)
                    try:
                        try:
                            io_buf = io.BytesIO()
                            plot.savefig(io_buf, format='raw')
                            plot.close('all')
                            io_buf.seek(0)
                            plots.append(Image.frombytes('RGBA', (1000, 400), io_buf.getvalue(), 'raw'))
                            io_buf.close()
                        except Exception:
                            warnings.warn("Byte IO fail!")
                            pass
                        if self.with_tracking:
                            comparison_plot = [wandb.Image(image, caption=f"Comparison of epoch {epoch+1} sample {idx}")
                                           for idx, image in enumerate(plots)]
                            self.wandb.log({"Examples": comparison_plot}, step=completed_step)
                    except Exception:
                        warnings.warn("Unable to log examples to wandb!")
                        pass

            if self.with_tracking:
                print("--- Logging to wandb ---")
                self.wandb.log({"Epoch": epoch+1,
                                "Loss_content_contributed": avg_epoch_content_loss*self.alpha,
                                "Loss_style_contributed": avg_epoch_style_loss*self.beta,
                                "Loss_variation_contributed": avg_epoch_var_loss*self.gamma,
                                "Loss_histogram_contributed": avg_epoch_hist_loss*self.delta,
                                "Total_loss": avg_epoch_total_loss,
                                "Min_loss": min(loss_list)}, step=completed_step)

            if avg_epoch_total_loss == min(loss_list) and self.save_best:
                print(f"\n Saving epoch [{epoch + 1}/{self.num_train_epochs}]")
                self.save(transformer, transformer_optimizer, decoder, info={"total": avg_epoch_total_loss,
                                                                    "content": avg_epoch_content_loss,
                                                                    "style": avg_epoch_style_loss,
                                                                    "epoch": epoch,
                                                                    "completed_step": completed_step
                                                                    }, result=plots)
            elif not self.save_best:
                print(f"\n Saving epoch [{epoch + 1}/{self.num_train_epochs}]")
                self.save(transformer, transformer_optimizer, decoder, info={"total": avg_epoch_total_loss,
                                                                    "content": avg_epoch_content_loss,
                                                                    "style": avg_epoch_style_loss,
                                                                    "epoch": epoch,
                                                                    "completed_step": completed_step
                                                                    }, result=plots)
            else:
                print(f"\n Discarding epoch [{epoch + 1}/{self.num_train_epochs}]")
                if self.plot_per_epoch:
                    plot.close('all')
                continue

        if self.with_tracking:
            self.wandb.finish()

    def save(self, transformer, transformer_optimizer, decoder=None, info=None, result=None):
        dir_name = f"TotalL_{info['total']}_Content_{info['content']}_Style_{info['style']}"
        save_path_dir = os.path.join(self.output_dir, dir_name)
        print(f" Saving in {save_path_dir}")
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir, exist_ok=False)
        else:
            warnings.warn("File exist!, might be a good idea to change seed")
            raise FileExistsError

        model_path = os.path.join(save_path_dir, f"transformer{len(self.dataloaders['train'].dataset)}" + ".pth")
        # Save weights at output dir
        torch.save({
            'epoch': info['epoch'],
            'model_state_dict': transformer.state_dict(),
            "decoder_state_dict": decoder.state_dict() if decoder is not None else None,
            'optimizer_state_dict': transformer_optimizer.state_dict(),
            "optim_name": self.optim_name,
            'loss': info['total'],
            'completed_step': info['completed_step'],
            "criterion": {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "delta": self.delta,
                "content_layers_idx": self.content_layers_idx,
                "style_layers_idx": self.style_layers_idx
            }
        }, model_path)

        try:
            config_path = os.path.join(save_path_dir,
                    f"config_E{info['epoch']}_{'_'.join(str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')).split())}")
            config_file = open(config_path, "w")
            config_file.write(f"\n          {PRJ_NAME}\n")
            config_file.write(f"\n   Epoch: {info['epoch']}")
            config_file.write(f"\n   Completed step: {info['completed_step']}")
            config_file.write(f"\n   Loss: {info['total']}")
            for key, value in vars(self.config).items():
                config_file.write(f"\n {key}: {value} ")
            config_file.close()
        except IOError:
            warnings.warn(f"Can't save config for this run {info['epoch']}")
            pass

        if result is not None:
            for idx, plot in enumerate(result):
                plot_path = os.path.join(save_path_dir, f"result_plot{idx}.png")
                plot.save(plot_path)

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

    @staticmethod
    def plot_comparison(encoder, decoder, transformer, content_img_url: list, style_img_url: list, device='cpu',
                        content_transformation=None, style_transformation=None, sleep: int = 5,
                        plot: bool = True, alpha: float=1.0):
        try:
            for content_url, style_url in zip(content_img_url, style_img_url):
                content_image = Image.open(content_url).convert('RGB')
                style_image = Image.open(style_url).convert('RGB')
        except IOError:
            raise "Invalid image url!"

        if content_transformation:
            content_image_tensor = content_transformation(content_image).to(device)

        org_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])
        content_image_tensor_org = org_transform(content_image)

        if style_transformation:
            style_image_tensor = style_transformation(style_image).to(device)

        transformer.eval()
        decoder.eval()

        encode_Cfeatures = encoder(content_image_tensor.unsqueeze(0))
        encode_Sfeatures = encoder(style_image_tensor.unsqueeze(0))

        transformed_features = transformer(encode_Cfeatures, encode_Sfeatures)
        transformed_features = alpha * transformed_features + (1 - alpha) * encode_Cfeatures

        decode_img = decoder(transformed_features)

        # Normalize output
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        decode_img = norm(decode_img)

        del encode_Cfeatures, encode_Sfeatures, transformed_features

        # Convert tensor images to numpy arrays and adjust their shape if needed
        image1_np = content_image_tensor_org.squeeze().permute(1, 2, 0).detach().cpu().numpy()  # Adjust dimensions as per your tensor shape
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

        return plt, decode_img
