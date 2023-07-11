import torch
import numpy as np
from typing import Tuple
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self, eps: float=1e-5):
        super(AdaIN, self).__init__()
        self.eps_limit = eps
        self.content_eps = nn.Parameter(torch.FloatTensor([1e-5]), requires_grad=True) # Trainable parameter
        self.style_eps = nn.Parameter(torch.FloatTensor([1e-5]), requires_grad=True)  # Trainable parameter

    @staticmethod
    def calc_mean_std(feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        assert 0 < eps < 1.0
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    @staticmethod
    def adjust_sigmoid(x, upper_limit=0.3):
        lower_limit = 1e-5
        x = torch.pow(torch.abs(x), 0.5)
        if x < lower_limit:
            scaled_output = lower_limit
        elif x > upper_limit:
            scaled_output = upper_limit
        else:
            scaled_output = x
        return scaled_output

    def get_current_eps(self):
        if torch.cuda.is_available():
            return [self.adjust_sigmoid(self.style_eps, self.eps_limit).item(),
                    self.adjust_sigmoid(self.content_eps, self.eps_limit).item()]
        else:
            return [self.adjust_sigmoid(self.style_eps, self.eps_limit),
                    self.adjust_sigmoid(self.content_eps, self.eps_limit)]

    def adaptive_instance_normalization(self, content_feat, style_feat, style_eps=1e-5, content_eps=1e-5):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat, eps=style_eps)
        content_mean, content_std = self.calc_mean_std(content_feat, eps=content_eps)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def forward(self, content_feat, style_feat):
        return self.adaptive_instance_normalization(content_feat, style_feat,
                                                    style_eps=self.adjust_sigmoid(self.style_eps, self.eps_limit),
                                                    content_eps=self.adjust_sigmoid(self.content_eps, self.eps_limit))












