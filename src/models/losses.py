import torch
import torch.nn as nn


# Define the Gram Matrix computation
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        features = input.view(b * c, h * w)
        gram_matrix = torch.mm(features, features.t())

        return gram_matrix.div(b * c * h * w)


def mse_loss(input, target):
    return torch.mean((input - target)**2)


def style_loss(style, combination):
    gram = GramMatrix()

    S = gram(style)
    C = gram(combination)

    channels = style.size(1)  # Assuming style and combination have the same number of channels
    size = style.size(2) * style.size(3)  # Assuming img_nrows and img_ncols are defined

    loss = torch.sum(torch.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))
    return loss


def total_variation_loss(image):
    # Calculate the total variation loss
    batch_size, channels, height, width = image.size()
    dx = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    dy = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
    tv_loss = torch.sum(dx) + torch.sum(dy)
    return tv_loss / (batch_size * channels * height * width)

