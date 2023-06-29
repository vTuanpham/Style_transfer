import sys
sys.path.insert(0,r'./') #Add root directory here
import os
import torch
import argparse
from random import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from src.models.generator import Encoder, Decoder
from src.models.transformer import MTranspose
from src.models.trainer import Trainer
from src.data.dataloader import STDataloader
from src.utils.image_plot import plot_image
import torchvision.transforms as transforms


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_save_cpkt', type=str, help="Path to the save directory json file")
    parser.add_argument('--do_compare', action='store_true', help="Whether to do comparison or not")
    parser.add_argument('--save_dir', action='store_true', help="Path to save dir")
    parser.add_argument('--path_to_content_dir', type=str, help="Path to content dir to stylized")
    parser.add_argument('--path_to_style_dir', type=str, help="Path to style dir to use as style")
    parser.add_argument('--interactive', action="store_true", help="Whether to enable interactive mode")
    parser.add_argument('--test_batch_size', type=int, default=6, help="Batch size of test dataloader")
    parser.add_argument('--max_test_samples', type=int, default=None, help="Sample size of the test dataset")
    parser.add_argument('--seed', type=int, default=42, help="Seed for dataloader shuffle")

    args = parser.parse_args(args)

    return args


# def main(args):
#     args = parse_args(args)





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = Encoder().eval().to(device)
decoder = Decoder().eval().to(device)
checkpoint = torch.load(r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\style_transfer\src\models\checkpoints\training_session\trans_size32\might_final\Checkpoint_Style_transfer_TotalL_3795.406985586824_Content_97.18574203172518_Style_5286.092544594127\transformer25000.pth")
transformer = MTranspose(matrix_size=checkpoint['trans_size'],
                         layer_depth=checkpoint['layer_depth'],
                         deep_learner=checkpoint['deep_learner'],
                         deep_dense=False).to(device)
transformer.load_state_dict(checkpoint['model_state_dict'])
transformer.eval()

_, result = Trainer.plot_comparison(encoder, decoder, transformer,
                                     [r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\style_transfer\src\data\eval_dir\content\8.jpg"],
                                    [r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\style_transfer\src\data\eval_dir\style\4.png"],
                                    transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.ToTensor()
                                    ]),
                                    device,
                                    sleep=20)

# trans = transforms.ToPILImage()
# result = trans(result.squeeze())
# result.save("src/result_img.jpg")


# if __name__ == "__main__":
#     main(sys.argv[1:])