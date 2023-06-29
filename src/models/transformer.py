import torch
import numpy as np
from typing import Tuple
import torch.nn as nn


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


class FFExtractor(nn.Module):
    def __init__(self, matrix_size=32, layer_depth=1, deep_learner=False, deep_dense=False):
        super(FFExtractor, self).__init__()
        self.layer_depth = layer_depth
        self.deep_learner = deep_learner
        self.deep_dense = deep_dense
        self.cnn_block_super_deep_transition = nn.Sequential(
                                                SymmetricPadding2D((1, 1, 1, 1)),
                                                nn.Conv2d(in_channels=256,
                                                          out_channels=512,
                                                          kernel_size=(3, 3),
                                                          stride=1, padding=0),
                                                nn.InstanceNorm2d(num_features=512, momentum=0.5, affine=True),
                                                nn.ReLU(inplace=True),
                                                SymmetricPadding2D((1, 1, 1, 1)),
                                                nn.Conv2d(in_channels=512,
                                                          out_channels=256,
                                                          kernel_size=(3, 3),
                                                          stride=1, padding=0),
                                                nn.InstanceNorm2d(num_features=256, momentum=0.5, affine=True),
                                                nn.ReLU(inplace=True),
                                            )
        self.cnn_block_deep = nn.Sequential(
                                SymmetricPadding2D((1, 1, 1, 1)),
                                nn.Conv2d(in_channels=256,
                                          out_channels=256,
                                          kernel_size=(3, 3),
                                          stride=1, padding=0),
                                nn.InstanceNorm2d(num_features=256, momentum=0.65, eps=0.3, affine=True),
                                nn.PReLU(num_parameters=1),
                                # nn.ReLU(inplace=True)
        )
        self.cnn_final = nn.Sequential(
                                 SymmetricPadding2D((1,1,1,1)),
                                 nn.Conv2d(in_channels=256,
                                           out_channels=128,
                                           kernel_size=(3,3),
                                           stride=1, padding=0),
                                 nn.InstanceNorm2d(num_features=128, momentum=0.65, eps=0.3, affine=True),
                                 nn.PReLU(num_parameters=1),
                                 # nn.ReLU(inplace=True),
                                 SymmetricPadding2D((1, 1, 1, 1)),
                                 nn.Conv2d(in_channels=128,
                                           out_channels=64,
                                           kernel_size=(3,3),
                                           stride=1, padding=0),
                                 nn.InstanceNorm2d(num_features=64, momentum=0.65, eps=0.3, affine=True),
                                 nn.PReLU(num_parameters=1),
                                 # nn.ReLU(inplace=True),
                                 SymmetricPadding2D((1, 1, 1, 1)),
                                 nn.Conv2d(in_channels=64,
                                           out_channels=matrix_size,
                                           kernel_size=(3,3),
                                           stride=1, padding=0),
                                 nn.InstanceNorm2d(num_features=matrix_size, momentum=0.65, eps=0.3, affine=True)
        )

        self.dense_deep = nn.Sequential(
                            nn.Linear(in_features=matrix_size*matrix_size, out_features=16*16),
                            nn.Linear(in_features=16*16, out_features=8*8),
                            nn.Linear(in_features=8*8, out_features=matrix_size*matrix_size),
        )

        self.dense = nn.Linear(in_features=matrix_size*matrix_size, out_features=matrix_size*matrix_size)

    def forward(self, x):
        out = x
        for idx, layer in enumerate(range(self.layer_depth)):
            out = self.cnn_block_deep(out)
        out = out.clone() + x
        if self.deep_learner:
            out = self.cnn_block_super_deep_transition(out)
            out = out.clone() + out
        out = self.cnn_final(out)
        b, c, h, w = out.size()
        out = out.view(b, c, -1)   # batch, channels, h*w
        out = torch.bmm(out, out.transpose(1, 2)).div(h * w) # Compute covariance matrix
        out = out.view(out.size(0), -1) # Flatten

        return self.dense_deep(out) if self.deep_dense else self.dense(out)


class MTranspose(nn.Module):
    def __init__(self, matrix_size=32,
                 layer_depth=1,
                 deep_learner = False,
                 deep_dense: bool = False):
        super(MTranspose, self).__init__()
        self.matrix_size = matrix_size
        self.compress = nn.Conv2d(in_channels=256,
                                  out_channels=matrix_size,
                                  kernel_size=(1,1),
                                  stride=1, padding=0)
        self.uncompress = nn.Conv2d(in_channels=matrix_size,
                                    out_channels=256,
                                    kernel_size=(1,1),
                                    stride=1, padding=0)

        self.style_FFE = FFExtractor(matrix_size=matrix_size,
                                     layer_depth=layer_depth,
                                     deep_learner=deep_learner,
                                     deep_dense=deep_dense)
        self.content_FFE = FFExtractor(matrix_size=matrix_size,
                                       layer_depth=layer_depth,
                                       deep_learner=deep_learner,
                                       deep_dense=False)

    def forward(self, content_features, style_features):
       cbatch, cchannels, cheight, cwidth = content_features.size()
       cF_reshaped = content_features.view(cbatch, cchannels, -1)
       cMean = torch.mean(cF_reshaped, dim=2, keepdim=True)
       cMean = cMean.unsqueeze(3)
       cMean = cMean.expand_as(content_features)
       content_features = content_features - cMean

       sbatch, schannels, sheight, swidth = style_features.size()
       sF_reshaped = style_features.view(sbatch, schannels, -1)
       sMean = torch.mean(sF_reshaped, dim=2, keepdim=True)
       sMean = sMean.unsqueeze(3)
       sMean_style = sMean.expand_as(style_features)
       style_features = style_features - sMean_style

       sMean_content = sMean.expand_as(content_features)

       compress_content = self.compress(content_features)
       b, c, h, w = compress_content.size()
       compress_content = compress_content.view(b, c, -1)


       cMatrix = self.content_FFE(content_features)
       sMatrix = self.style_FFE(style_features)

       sMatrix = sMatrix.view(sMatrix.size(0), self.matrix_size, self.matrix_size)
       cMatrix = cMatrix.view(cMatrix.size(0), self.matrix_size, self.matrix_size)

       TMatrix = torch.bmm(sMatrix, cMatrix)
       TFeature = torch.bmm(TMatrix, compress_content).view(b, c, h, w)

       output = self.uncompress(TFeature)
       output += sMean_content

       return output












