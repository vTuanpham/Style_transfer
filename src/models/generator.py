import sys
sys.path.insert(0,r'./')
import numpy as np
from typing import Tuple
from functools import reduce
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


# Custom layer inherited from nn.Module
class SymmetricPadding2D(nn.Module):
    def __init__(self, padding: Tuple[int, int, int, int] = (1, 1, 1, 1)):  # left, right, top, bottom
        super(SymmetricPadding2D, self).__init__()
        self.padding = padding

    def forward(self, im: torch.Tensor):
        h, w = im.shape[-2:]
        left, right, top, bottom = self.padding

        x_idx = np.arange(-left, w + right)
        y_idx = np.arange(-top, h + bottom)

        def reflect(x, minx, maxx):
            """ Reflects an array around two points making a triangular waveform that ramps up
            and down, allowing for pad lengths greater than the input length """
            rng = maxx - minx
            double_rng = 2 * rng
            mod = np.fmod(x - minx, double_rng)
            normed_mod = np.where(mod < 0, mod + double_rng, mod)
            out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
            return np.array(out, dtype=x.dtype)

        x_pad = reflect(x_idx, -0.5, w - 0.5)
        y_pad = reflect(y_idx, -0.5, h - 0.5)
        xx, yy = np.meshgrid(x_pad, y_pad)
        return im[..., yy, xx]


class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.encode = nn.Sequential( # Sequential,
			nn.Conv2d(3,3,(1, 1)),
			SymmetricPadding2D((1, 1, 1, 1)),
			nn.Conv2d(3,64,(3, 3)),
			nn.ReLU(),
			SymmetricPadding2D((1, 1, 1, 1)),
			nn.Conv2d(64,64,(3, 3)),
			nn.ReLU(),
			nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
			SymmetricPadding2D((1, 1, 1, 1)),
			nn.Conv2d(64,128,(3, 3)),
			nn.ReLU(),
			SymmetricPadding2D((1, 1, 1, 1)),
			nn.Conv2d(128,128,(3, 3)),
			nn.ReLU(),
			nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
			SymmetricPadding2D((1, 1, 1, 1)),
			nn.Conv2d(128,256,(3, 3)),
			nn.ReLU(),
		)
		self.encode.load_state_dict(torch.load(r'./src/models/checkpoints/WCT_encoder_decoder/vgg_normalised_conv3_1.pth'))

	def forward(self, x):
		out = self.encode(x)
		return out


class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.decoder = nn.Sequential(  # Sequential,
			SymmetricPadding2D((1, 1, 1, 1)),
			nn.Conv2d(256, 128, (3, 3)),
			nn.ReLU(),
			nn.UpsamplingNearest2d(scale_factor=2),
			SymmetricPadding2D((1, 1, 1, 1)),
			nn.Conv2d(128, 128, (3, 3)),
			nn.ReLU(),
			SymmetricPadding2D((1, 1, 1, 1)),
			nn.Conv2d(128, 64, (3, 3)),
			nn.ReLU(),
			nn.UpsamplingNearest2d(scale_factor=2),
			SymmetricPadding2D((1, 1, 1, 1)),
			nn.Conv2d(64, 64, (3, 3)),
			nn.ReLU(),
			SymmetricPadding2D((1, 1, 1, 1)),
			nn.Conv2d(64, 3, (3, 3)),
		)
		self.decoder.load_state_dict(torch.load(r"./src/models/checkpoints/WCT_encoder_decoder/feature_invertor_conv3_1.pth"))

	def forward(self, x):
		out = self.decoder(x)
		return out


if __name__ == "__main__":
	vgg = Encoder()
	dec = Decoder()

	vgg.eval()
	dec.eval()
	# Load and preprocess the input image
	image_path = r'./src/data/dummy/content/im1.jpg'  # Replace with your image path
	image = Image.open(image_path)
	preprocess = transforms.Compose([
		# transforms.Resize((256, 256)),  # Resize to match the input size of the model
		transforms.ToTensor(),  # Convert PIL image to tensor
	])
	input_tensor = preprocess(image)
	input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

	# Run the encoder and decoder
	with torch.no_grad():
		encoded_features = vgg(input_batch)
		reconstructed_image = dec(encoded_features)

	# Convert the output tensor to a PIL image
	output_image = transforms.ToPILImage()(reconstructed_image.squeeze(0).cpu())

	# Display the input and reconstructed images
	image.show(title='Input Image')
	output_image.show(title='Reconstructed Image')