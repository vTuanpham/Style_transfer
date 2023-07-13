import sys
sys.path.insert(0,r'./') #Add root directory here
import os
import io
import torch
import argparse
from random import random
from PIL import Image
from collections import OrderedDict

import numpy as np
import torchvision.transforms as transforms

from src.models.generator import Encoder, Decoder
from src.models.transformer import AdaIN
from src.models.trainer import Trainer
from src.data.dataloader import STDataloader
from src.utils.custom_transform import RGBToGrayscaleStacked
from src.utils.utils import timeit


def parse_args(args):
    parser = argparse.ArgumentParser(description="Inference from the model, plot and save result")

    parser.add_argument('-cpkt', '--path_to_save_cpkt', type=str, help="Path to the save checkpoint")
    parser.add_argument('-r', '--res', type=int, default=450, help="Inference resolution")
    parser.add_argument('--plot_time', type=int, default=40, help="Comparison plot time")
    parser.add_argument('--do_plot', action='store_true', default=True, help="Whether to plot or not")
    parser.add_argument('--save_dir', type=str, default='./src/output', help="Path to save dir")
    parser.add_argument('-c', '--path_to_content', type=str, help="Path to content dir to stylized")
    parser.add_argument('-s', '--path_to_style', type=str, help="Path to style dir to use as style")
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help="alpha value for style and content adjustment")

    args = parser.parse_args(args)

    assert os.path.isfile(args.path_to_save_cpkt), "Please provide the .pth file"
    assert os.path.isfile(args.path_to_content), "Please provide the correct content image path"
    assert os.path.isfile(args.path_to_style), "Please provide the correct style image path"
    assert 64 < args.res < 1500, "Resolution too small or too large!"
    assert args.plot_time > 1, "Please provide the plot time larger than 0"

    return args


@timeit
def main(args):
    args = parse_args(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    checkpoint = torch.load(args.path_to_save_cpkt)
    transformer = AdaIN().to(device)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    transformer.load_state_dict(checkpoint['model_state_dict'])
    decoder.eval()
    transformer.eval()

    plot, result = Trainer.plot_comparison(encoder, decoder, transformer,
                                         [args.path_to_content],
                                        [args.path_to_style],
                                        content_transformation=transforms.Compose([
                                            transforms.Resize(args.res),
                                            transforms.ToTensor(),
                                            RGBToGrayscaleStacked(False)
                                        ]),
                                        style_transformation=transforms.Compose([
                                            transforms.Resize(args.res),
                                            transforms.ToTensor()
                                        ]),
                                        device=device,
                                        sleep=args.plot_time, alpha=args.alpha, plot=False)

    image3_np = result.squeeze().permute(1, 2, 0).detach().cpu() # Adjust dimensions as per your tensor shape
    image3_np = np.interp(image3_np, (image3_np.min(), image3_np.max()), (0, 1))
    result = Image.fromarray((image3_np * 255).astype(np.uint8))

    try:
        io_buf = io.BytesIO()
        plot.savefig(io_buf, format='raw')
        plot.close('all')
        io_buf.seek(0)
        result_plot = Image.frombytes('RGBA', (1000, 400), io_buf.getvalue(), 'raw')
        plot.close('all')
        io_buf.close()
    except Exception:
        raise "Byte IO fail!"

    result_plot.show()
    result_plot.save(os.path.join(args.save_dir, f"plot_A{args.alpha}.png"))
    result.save(os.path.join(args.save_dir, f"result_A{args.alpha}.png"))


if __name__ == "__main__":
    main(sys.argv[1:])