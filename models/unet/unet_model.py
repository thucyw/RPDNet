""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.inc = DoubleConv(n_channels, 64)
        self.inc = nn.Conv2d(3, 64, kernel_size=1)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    # def temporal_shift(self, x):
    #     _, c, h, w = x.size()
    #     x = x.view(-1,self.T,c,h,w)
    #     fold = c//16
    #     out = torch.zeros_like(x)
    #     out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
    #     out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
    #     out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
    #     out = out.view(-1, c, h, w)
        
    #     return out

    def forward(self, x):
        B, C, H, W = x.size()
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().view(1, 1, 1, W).repeat(B, 1, H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().view(1, 1, H, 1).repeat(B, 1, 1, W)
        x = torch.cat((loc_w, loc_h, x), 1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # x2 = self.temporal_shift(x2)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x4 = self.temporal_shift(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # x = self.temporal_shift(x)
        logits = self.outc(x)
        
        return logits

if __name__ == "__main__":
    model = UNet(1, 1)
    x = torch.rand(1, 1, 512, 512)
    out = model(x)
    print(out.shape)
