import torch
import torch.nn as nn
import torch.nn.functional as F


class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        f = input.view(b,c,h*w) # bxcx(hxw)
        # torch.bmm(batch1, batch2, out=None)   #
        # batch1: bxmxp, batch2: bxpxn -> bxmxn #
        G = torch.bmm(f,f.transpose(1,2)) # f: bxcx(hxw), f.transpose: bx(hxw)xc -> bxcxc
        return G.div_(c*h*w)


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.G = GramMatrix()
        self.mse_loss = nn.MSELoss(size_average=False)

    def forward(self, input, target):
        ib,ic,ih,iw = input.size()
        iF = input.view(ib,ic,-1)
        iMean = torch.mean(iF,dim=2)
        iCov = self.G(input)

        tb,tc,th,tw = target.size()
        tF = target.view(tb,tc,-1)
        tMean = torch.mean(tF,dim=2)
        tCov = self.G(target)

        loss = self.mse_loss(iMean,tMean) + self.mse_loss(iCov,tCov)
        return loss/tb


class TVLoss(nn.Module):
    def forward(self, image):
        batch_size, channels, height, width = image.size()
        count_h = self._tensor_size(image[:, :, 1:, :])
        count_w = self._tensor_size(image[:, :, :, 1:])
        h_tv = torch.pow((image[:, :, 1:, :] - image[:, :, :height - 1, :]), 2).div(count_h)
        w_tv = torch.pow((image[:, :, :, 1:] - image[:, :, :, :width - 1]), 2).div(count_w)
        tv_loss = torch.sum(h_tv) + torch.sum(w_tv)
        return 2 * tv_loss / batch_size

    @staticmethod
    def _tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


