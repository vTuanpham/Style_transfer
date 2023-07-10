import sys
sys.path.insert(0,r'./') #Add root directory here
import os
import torch
import argparse
from random import random
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict

import numpy as np
from src.models.generator import Encoder, Decoder
from src.models.transformer import AdaIN
from src.models.trainer import Trainer
from src.data.dataloader import STDataloader
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
    parser.add_argument('--alpha', type=float, default=1.0, help="alpha value for style and content adjustment")

    args = parser.parse_args(args)

    return args


def main(args):
    args = parse_args(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    checkpoint = torch.load(args.path_to_save_cpkt)
    decoder_cpkt = torch.load(r'C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\style_transfer\src\models\checkpoints\training_session\AdaIN\decoder.pth')

    transformer = AdaIN().to(device)
    decoder.decoder.load_state_dict(decoder_cpkt)
    transformer.load_state_dict(checkpoint['model_state_dict'])
    decoder.eval()
    transformer.eval()

    _, result = Trainer.plot_comparison(encoder, decoder, transformer,
                                         [r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\style_transfer\src\data\359733373_638860051290066_4793153396181217139_n.png"],
                                        [r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\style_transfer\src\data\painting.jpg"],
                                        transforms.Compose([
                                            transforms.Resize(450),
                                            transforms.ToTensor()
                                        ]),
                                        device,
                                        sleep=20, alpha=args.alpha)

    image3_np = result.squeeze().permute(1, 2, 0).detach().cpu() # Adjust dimensions as per your tensor shape
    image3_np = np.interp(image3_np, (image3_np.min(), image3_np.max()), (0, 1))

    result = Image.fromarray((image3_np * 255).astype(np.uint8))

    result.save("src/result_img.jpg")


if __name__ == "__main__":
    main(sys.argv[1:])