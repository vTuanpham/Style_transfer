import time
import warnings

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class AdaINStyleLoss(nn.Module):
    def __init__(self):
        super(AdaINStyleLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    @staticmethod
    def calc_mean_std(feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        # assert (target.requires_grad is False)
        input_mean, input_std = self.calc_mean_std(input)
        target_mean, target_std = self.calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, input, target):
        return self.calc_style_loss(input, target)


class VincentStyleLoss(nn.Module):
    def __init__(self, scale_factor=1e-5):
        super(VincentStyleLoss, self).__init__()
        warnings.warn("\n Using VincentStyleLoss might me slow!")
        self.scale_factor = scale_factor  # Defaults tend to be very large, we scale to make them easier to work with

    @staticmethod
    def calc_2_moments(x):
        b, c, w, h = x.shape
        x = x.reshape(b, c, w * h)  # b, c, n
        mu = x.mean(dim=-1, keepdim=True)  # b, c, 1
        cov = torch.bmm(x - mu, torch.transpose(x - mu, -1, -2))
        return mu, cov

    @staticmethod
    def matrix_diag(diagonal):
        N = diagonal.shape[-1]
        shape = diagonal.shape[:-1] + (N, N)
        device, dtype = diagonal.device, diagonal.dtype
        result = torch.zeros(shape, dtype=dtype, device=device)
        indices = torch.arange(result.numel(), device=device).reshape(shape)
        indices = indices.diagonal(dim1=-2, dim2=-1)
        result.view(-1)[indices] = diagonal
        return result

    def l2wass_dist(self, mean_stl, cov_stl, mean_synth, cov_synth):
        # Calculate tr_cov and root_cov from mean_stl and cov_stl
        eigvals, eigvects = torch.linalg.eigh(
            cov_stl)  # eig returns complex tensors, I think eigh matches tf self_adjoint_eig
        eigroot_mat = self.matrix_diag(torch.sqrt(eigvals.clip(0)))
        root_cov_stl = torch.matmul(torch.matmul(eigvects, eigroot_mat), torch.transpose(eigvects, -1, -2))
        tr_cov_stl = torch.sum(eigvals.clip(0), dim=1, keepdim=True)

        tr_cov_synth = torch.sum(torch.linalg.eigvalsh(cov_synth).clip(0), dim=1, keepdim=True)
        mean_diff_squared = torch.mean((mean_synth - mean_stl) ** 2)
        cov_prod = torch.matmul(torch.matmul(root_cov_stl, cov_synth), root_cov_stl)
        var_overlap = torch.sum(torch.sqrt(torch.linalg.eigvalsh(cov_prod).clip(0.1)), dim=1,
                                keepdim=True)  # .clip(0) meant errors getting eigvals
        dist = mean_diff_squared + tr_cov_stl + tr_cov_synth - 2 * var_overlap
        return dist

    def forward(self, input_f, target_f):

        mean_synth, cov_synth = self.calc_2_moments(input_f)  # input mean and cov
        mean_stl, cov_stl = self.calc_2_moments(target_f)  # target mean and cov
        loss = self.l2wass_dist(mean_stl, cov_stl, mean_synth, cov_synth)
        del mean_stl, cov_stl, mean_synth, cov_synth, input_f, target_f

        return loss.mean() * self.scale_factor


class AdaINContentLoss(nn.Module):
    def __init__(self):
        super(AdaINContentLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        assert (input.size() == target.size())
        # assert (target.requires_grad is False)
        return self.mse_loss(input, target)


class TVLoss(nn.Module):
    def forward(self, image):
        batch_size, channels, height, width = image.size()
        count_h = self._tensor_size(image[:, :, 1:, :])
        count_w = self._tensor_size(image[:, :, :, 1:])
        h_tv = torch.pow((image[:, :, 1:, :] - image[:, :, :height - 1, :]), 2).div(count_h)
        w_tv = torch.pow((image[:, :, :, 1:] - image[:, :, :, :width - 1]), 2).div(count_w)
        tv_loss = torch.sum(h_tv) + torch.sum(w_tv)
        return (2 * tv_loss / batch_size)*100

    @staticmethod
    def _tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


EPS = 1e-6


class RGBuvHistBlock(nn.Module):
    def __init__(self, h=64, insz=150, resizing='interpolation',
                 method='inverse-quadratic', sigma=0.02, intensity_scale=True,
                 hist_boundary=None, green_only=False, device='cuda'):
        """ Computes the RGB-uv histogram feature of a given image.
        Args:
          h: histogram dimension size (scalar). The default value is 64.
          insz: maximum size of the input image; if it is larger than this size, the
            image will be resized (scalar). Default value is 150 (i.e., 150 x 150
            pixels).
          resizing: resizing method if applicable. Options are: 'interpolation' or
            'sampling'. Default is 'interpolation'.
          method: the method used to count the number of pixels for each bin in the
            histogram feature. Options are: 'thresholding', 'RBF' (radial basis
            function), or 'inverse-quadratic'. Default value is 'inverse-quadratic'.
          sigma: if the method value is 'RBF' or 'inverse-quadratic', then this is
            the sigma parameter of the kernel function. The default value is 0.02.
          intensity_scale: boolean variable to use the intensity scale (I_y in
            Equation 2). Default value is True.
          hist_boundary: a list of histogram boundary values. Default is [-3, 3].
          green_only: boolean variable to use only the log(g/r), log(g/b) channels.
            Default is False.

        Methods:
          forward: accepts input image and returns its histogram feature. Note that
            unless the method is 'thresholding', this is a differentiable function
            and can be easily integrated with the loss function. As mentioned in the
             paper, the 'inverse-quadratic' was found more stable than 'RBF' in our
             training.
        """
        super(RGBuvHistBlock, self).__init__()
        self.h = h
        self.insz = insz
        self.device = device
        self.resizing = resizing
        self.method = method
        self.intensity_scale = intensity_scale
        self.green_only = green_only
        if hist_boundary is None:
            hist_boundary = [-3, 3]
        hist_boundary.sort()
        self.hist_boundary = hist_boundary
        if self.method == 'thresholding':
            self.eps = (abs(hist_boundary[0]) + abs(hist_boundary[1])) / h
        else:
            self.sigma = sigma

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        if x.shape[2] > self.insz or x.shape[3] > self.insz:
            if self.resizing == 'interpolation':
                x_sampled = F.interpolate(x, size=(self.insz, self.insz),
                                          mode='bilinear', align_corners=False)
            elif self.resizing == 'sampling':
                inds_1 = torch.LongTensor(
                    np.linspace(0, x.shape[2], self.h, endpoint=False)).to(
                    device=self.device)
                inds_2 = torch.LongTensor(
                    np.linspace(0, x.shape[3], self.h, endpoint=False)).to(
                    device=self.device)
                x_sampled = x.index_select(2, inds_1)
                x_sampled = x_sampled.index_select(3, inds_2)
            else:
                raise Exception(
                    f'Wrong resizing method. It should be: interpolation or sampling. '
                    f'But the given value is {self.resizing}.')
        else:
            x_sampled = x

        L = x_sampled.shape[0]  # size of mini-batch
        if x_sampled.shape[1] > 3:
            x_sampled = x_sampled[:, :3, :, :]
        X = torch.unbind(x_sampled, dim=0)
        hists = torch.zeros((x_sampled.shape[0], 1 + int(not self.green_only) * 2,
                             self.h, self.h)).to(device=self.device)
        for l in range(L):
            I = torch.t(torch.reshape(X[l], (3, -1)))
            II = torch.pow(I, 2)
            if self.intensity_scale:
                Iy = torch.unsqueeze(torch.sqrt(II[:, 0] + II[:, 1] + II[:, 2] + EPS),
                                     dim=1)
            else:
                Iy = 1
            if not self.green_only:
                Iu0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 1] +
                                                                           EPS), dim=1)
                Iv0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 2] +
                                                                           EPS), dim=1)
                diff_u0 = abs(
                    Iu0 - torch.unsqueeze(torch.tensor(np.linspace(
                        self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                        dim=0).to(self.device))
                diff_v0 = abs(
                    Iv0 - torch.unsqueeze(torch.tensor(np.linspace(
                        self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                        dim=0).to(self.device))
                if self.method == 'thresholding':
                    diff_u0 = torch.reshape(diff_u0, (-1, self.h)) <= self.eps / 2
                    diff_v0 = torch.reshape(diff_v0, (-1, self.h)) <= self.eps / 2
                elif self.method == 'RBF':
                    diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_u0 = torch.exp(-diff_u0)  # Radial basis function
                    diff_v0 = torch.exp(-diff_v0)
                elif self.method == 'inverse-quadratic':
                    diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_u0 = 1 / (1 + diff_u0)  # Inverse quadratic
                    diff_v0 = 1 / (1 + diff_v0)
                else:
                    raise Exception(
                        f'Wrong kernel method. It should be either thresholding, RBF,'
                        f' inverse-quadratic. But the given value is {self.method}.')
                diff_u0 = diff_u0.type(torch.float32)
                diff_v0 = diff_v0.type(torch.float32)
                a = torch.t(Iy * diff_u0)
                hists[l, 0, :, :] = torch.mm(a, diff_v0)

            Iu1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 0] + EPS),
                                  dim=1)
            Iv1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 2] + EPS),
                                  dim=1)
            diff_u1 = abs(
                Iu1 - torch.unsqueeze(torch.tensor(np.linspace(
                    self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                    dim=0).to(self.device))
            diff_v1 = abs(
                Iv1 - torch.unsqueeze(torch.tensor(np.linspace(
                    self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                    dim=0).to(self.device))

            if self.method == 'thresholding':
                diff_u1 = torch.reshape(diff_u1, (-1, self.h)) <= self.eps / 2
                diff_v1 = torch.reshape(diff_v1, (-1, self.h)) <= self.eps / 2
            elif self.method == 'RBF':
                diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u1 = torch.exp(-diff_u1)  # Gaussian
                diff_v1 = torch.exp(-diff_v1)
            elif self.method == 'inverse-quadratic':
                diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u1 = 1 / (1 + diff_u1)  # Inverse quadratic
                diff_v1 = 1 / (1 + diff_v1)

            diff_u1 = diff_u1.type(torch.float32)
            diff_v1 = diff_v1.type(torch.float32)
            a = torch.t(Iy * diff_u1)
            if not self.green_only:
                hists[l, 1, :, :] = torch.mm(a, diff_v1)
            else:
                hists[l, 0, :, :] = torch.mm(a, diff_v1)

            if not self.green_only:
                Iu2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 0] +
                                                                           EPS), dim=1)
                Iv2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 1] +
                                                                           EPS), dim=1)
                diff_u2 = abs(
                    Iu2 - torch.unsqueeze(torch.tensor(np.linspace(
                        self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                        dim=0).to(self.device))
                diff_v2 = abs(
                    Iv2 - torch.unsqueeze(torch.tensor(np.linspace(
                        self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                        dim=0).to(self.device))
                if self.method == 'thresholding':
                    diff_u2 = torch.reshape(diff_u2, (-1, self.h)) <= self.eps / 2
                    diff_v2 = torch.reshape(diff_v2, (-1, self.h)) <= self.eps / 2
                elif self.method == 'RBF':
                    diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_u2 = torch.exp(-diff_u2)  # Gaussian
                    diff_v2 = torch.exp(-diff_v2)
                elif self.method == 'inverse-quadratic':
                    diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_u2 = 1 / (1 + diff_u2)  # Inverse quadratic
                    diff_v2 = 1 / (1 + diff_v2)
                diff_u2 = diff_u2.type(torch.float32)
                diff_v2 = diff_v2.type(torch.float32)
                a = torch.t(Iy * diff_u2)
                hists[l, 2, :, :] = torch.mm(a, diff_v2)

        # normalization
        hists_normalized = hists / (
                ((hists.sum(dim=1)).sum(dim=1)).sum(dim=1).view(-1, 1, 1, 1) + EPS)

        return hists_normalized


class HistLoss(nn.Module):
    def __init__(self, intensity_scale=True,
                 histogram_size=64,
                 max_input_size=128,
                 hist_boundary=[-3, 3],
                 method = 'inverse-quadratic'):   # options:'thresholding','RBF','inverse-quadratic'
        super(HistLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.histogram_block = RGBuvHistBlock(insz=max_input_size, h=histogram_size,
                                         intensity_scale=intensity_scale,
                                         method=method, hist_boundary=hist_boundary,
                                         device=device)

    def forward(self, input, target):
        input_hist = self.histogram_block(input)
        target_hist = self.histogram_block(target)

        histogram_loss = (1 / np.sqrt(2.0) *
                          (torch.sqrt(torch.sum(torch.pow(torch.sqrt(target_hist) -
                                                torch.sqrt(input_hist), 2)))) /input_hist.shape[0])
        del input, target, input_hist, target_hist

        return histogram_loss*100


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = transforms.Compose(
        [transforms.ToTensor()])  # transform it into a torch tensor

    unloader = transforms.ToPILImage()  # reconvert into PIL image


    def image_loader(image_name):
        image = Image.open(image_name)
        # fake batch dimension required to fit network's input dimensions
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)


    # read images
    input_image = image_loader(r"./src/data/eval_dir/content/1.jpg")

    target_image = image_loader(r"./src/data/eval_dir/style/1.jpg")


    def imshow(tensor, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(5)  # pause a bit so that plots are updated


    plt.ion()

    plt.figure()
    imshow(input_image, title='Input Image')

    plt.figure()
    imshow(target_image, title='Target Image')

    intensity_scale = True

    histogram_size = 64

    max_input_size = 256

    hist_boundary = [-3, 3]

    method = 'inverse-quadratic'  # options:'thresholding','RBF','inverse-quadratic'

    # create a histogram block
    histogram_block = RGBuvHistBlock(insz=max_input_size, h=histogram_size,
                                     intensity_scale=intensity_scale,
                                     method=method, hist_boundary=hist_boundary,
                                     device=device)

    input_hist = histogram_block(input_image)
    target_hist = histogram_block(target_image)

    plt.ion()

    plt.figure()
    imshow(input_hist * 100, title='Input Histogram')

    plt.figure()
    imshow(target_hist * 100, title='Target Histogram')

    histogram_loss = (1 / np.sqrt(2.0) * (torch.sqrt(torch.sum(
        torch.pow(torch.sqrt(target_hist) - torch.sqrt(input_hist), 2)))) /
                      input_hist.shape[0])

    print(f'Histogram loss = {histogram_loss}')
    time.sleep(20)





