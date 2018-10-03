import torch.nn as nn
import numpy as np
from .base_network import NetworkBase, Crop
import torch

class BeganDiscriminatorv3(NetworkBase):
	"""docstring for ClassName"""
	def __init__(self, in_channel_num=3, out_channel_num=3):
		super(BeganDiscriminatorv3, self).__init__()
		self._name = 'began_discriminator_v3'

		layers = []
		layers.append(nn.Conv2d(in_channel_num, 32, 3, 1, 1, bias=False)) # bnx3x128x128 -> bnx32x128x128
		layers.append(nn.BatchNorm2d(32))
		layers.append(nn.ELU())

		layers.append(nn.Conv2d(32, 64, 3, 1, 1, bias=False))            # bnx32x128x128 -> bnx64x128x128
		layers.append(nn.BatchNorm2d(64))
		layers.append(nn.ELU())

		layers.append(nn.ZeroPad2d((0, 1, 0, 1)))                        # bnx64x128x128 -> bnx64x129x129
		layers.append(nn.Conv2d(64, 64, 3, 2, 0, bias=False))            # bnx64x129x129 -> bnx64x48x48
		layers.append(nn.BatchNorm2d(64))
		layers.append(nn.ELU())

		layers.append(nn.Conv2d(64, 64, 3, 1, 1, bias=False))            # bnx64x64x64 -> bnx64x64x64
		layers.append(nn.BatchNorm2d(64))
		layers.append(nn.ELU())

		layers.append(nn.Conv2d(64, 128, 3, 1, 1, bias=False))           # bnx64x64x64 -> bnx128x64x64
		layers.append(nn.BatchNorm2d(128))
		layers.append(nn.ELU())

		layers.append(nn.ZeroPad2d((0, 1, 0, 1)))                        # bnx128x64x64 -> bnx128x65x65
		layers.append(nn.Conv2d(128, 128, 3, 2, 0, bias=False))          # bnx128x64x64 -> bnx128x24x24
		layers.append(nn.BatchNorm2d(128))
		layers.append(nn.ELU())

		layers.append(nn.Conv2d(128, 128, 3, 1, 1, bias=False))          # bnx128x32x32 -> bnx128x32x32
		layers.append(nn.BatchNorm2d(128))
		layers.append(nn.ELU())

		layers.append(nn.Conv2d(128, 128, 3, 1, 1, bias=False))          # bnx128x32x32 -> bnx128x32x32
		layers.append(nn.BatchNorm2d(128))
		layers.append(nn.ELU())

		self.encoder = nn.Sequential(*layers)

		self.fc1 = nn.Linear(128*24*24, 512)
		self.fc2 = nn.Linear(512, 128*24*24)

		layers_2 = []
		layers_2.append(nn.Conv2d(128, 128, 3, 1, 1, bias=False))          # bnx128x32x32 -> bnx128x32x32
		layers_2.append(nn.BatchNorm2d(128))
		layers_2.append(nn.ELU())

		layers_2.append(nn.Conv2d(128, 128, 3, 1, 1, bias=False))          # bnx128x32x32 -> bnx128x32x32
		layers_2.append(nn.BatchNorm2d(128))
		layers_2.append(nn.ELU())

		layers_2.append(nn.ConvTranspose2d(128, 128, 3, 2, 0, bias=False))          # bnx128x32x32 -> bnx128x65x65
		layers_2.append(nn.BatchNorm2d(128))
		layers_2.append(nn.ELU())
		layers_2.append(Crop([0, 1, 0, 1]))                                         # bnx128x65x65 -> bnx128x64x64

		layers_2.append(nn.Conv2d(128, 64, 3, 1, 1, bias=False))           # bnx128x64x64 -> bnx64x64x64
		layers_2.append(nn.BatchNorm2d(64))
		layers_2.append(nn.ELU())

		layers_2.append(nn.Conv2d(64, 64, 3, 1, 1, bias=False))           # bnx64x64x64 -> bnx64x64x64
		layers_2.append(nn.BatchNorm2d(64))
		layers_2.append(nn.ELU())

		layers_2.append(nn.ConvTranspose2d(64, 64, 3, 2, 0, bias=False))          # bnx64x64x64 -> bnx64x129x129
		layers_2.append(nn.BatchNorm2d(64))
		layers_2.append(nn.ELU())
		layers_2.append(Crop([0, 1, 0, 1]))                                         # bnx64x129x129 -> bnx64x128x128

		layers_2.append(nn.Conv2d(64, 32, 3, 1, 1, bias=False))           # bnx64x128x128 -> bnx32x128x128
		layers_2.append(nn.BatchNorm2d(32))
		layers_2.append(nn.ELU())

		layers_2.append(nn.Conv2d(32, 32, 3, 1, 1, bias=False))           # bnx32x128x128 -> bnx32x128x128
		layers_2.append(nn.BatchNorm2d(32))
		layers_2.append(nn.ELU())

		layers_2.append(nn.Conv2d(32, out_channel_num, 3, 1, 1, bias=False))           # bnx32x128x128 -> bnxout_chx128x128
		#layers_2.append(nn.Tanh()) # ???? need????
		self.decoder = nn.Sequential(*layers_2)

	def forward(self, input_img, input_heatmap):
		# print(input_img.size())
		# print(input_heatmap.size())
		# input_heatmap = input_heatmap.view(-1,1,128,128)
		# input = torch.cat([input_img, input_heatmap], dim=1)
		input = input_img

		x = self.encoder(input)

		x = x.view(-1, 128*24*24)

		x = self.fc1(x)

		x = x.view(-1, 512)

		x = self.fc2(x)

		x = x.view(-1, 128, 24, 24)

		x = self.decoder(x)

		#print(x.size())

		return x

