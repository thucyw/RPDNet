import torch
import torch.nn as nn
import torch.nn.functional as F

class fcn(nn.Module):

    def __init__(self):
        super(fcn, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer6 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer7 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        B, _, H, W = x.size()
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().view(1, 1, 1, W).repeat(B, 1, H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().view(1, 1, H, 1).repeat(B, 1, 1, W)
        x = torch.cat((loc_w, loc_h, x), 1)
        feat = []
        for i in range(8):
            x_ = getattr(self, 'layer{}'.format(i))(x)
            feat.append(nn.Upsample(scale_factor=2**i, mode='nearest')(x_))
            x = x_

        x = torch.cat(feat, 1)

        out = self.layer8(x)

        return out