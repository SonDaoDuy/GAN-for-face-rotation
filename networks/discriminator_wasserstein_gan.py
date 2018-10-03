import torch.nn as nn
import numpy as np
from .base_network import NetworkBase

class DiscriminatorWGAN(NetworkBase):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=96, conv_dim=64, c_dim=10, repeat_num=6):
        super(DiscriminatorWGAN, self).__init__()
        self._name = 'discriminator_wgan'
        self.feature = []

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 2*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

    def forward(self, x, heatmap):
        h = self.main(x)
        self.feature = h.view(-1, h.size()[1] * h.size()[2] * h.size()[3])
        out_real = self.conv1(h)
        #print(out_real.size()) #suppose to be a matrix of 1 and 0
        out_aux = self.conv2(h)
        #print(out_aux.size()) # suppose to be a vector of 10?????
        return out_real.squeeze(), out_aux.squeeze(), self.feature