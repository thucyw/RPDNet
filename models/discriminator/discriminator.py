import torch
import torch.nn as nn
from math import sqrt
from utils.function import BinarizedF

class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.binarize = BinarizedF()

        self.main_module = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True))

        # self._initialize_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()


    def forward(self, x1, x2):
        
        x = x1*self.binarize.forward(x2)
        x = self.main_module(x)

        return x