import torch.nn as nn
import numpy as np
from .base_network import NetworkBase, Crop
import torch

class CasiaGeneratorv3(NetworkBase):
	"""

	"""
	def __init__(self, in_channel_num=1, out_channel_num=3):
		super(CasiaGeneratorv3, self).__init__()
		self._name = 'casia_generator_v3'
		self.features = []

		#Encoder
		G_enc_convLayers = [
			nn.Conv2d(in_channel_num + 3, 32, 3, 1, 1, bias=False), # Bx3x128x128 -> Bx32x128x128
			nn.BatchNorm2d(32),
			nn.ELU(),
			nn.Conv2d(32, 64, 3, 1, 1, bias=False), # Bx32x128x128 -> Bx64x128x128
			nn.BatchNorm2d(64),
			nn.ELU(),
			nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx64x128x128 -> Bx64x129x129
			nn.Conv2d(64, 64, 3, 2, 0, bias=False), # Bx64x129x129 -> Bx64x64x64
			nn.BatchNorm2d(64),
			nn.ELU(),
			nn.Conv2d(64, 64, 3, 1, 1, bias=False), # Bx64x64x64 -> Bx64x64x64
			nn.BatchNorm2d(64),
			nn.ELU(),
			nn.Conv2d(64, 128, 3, 1, 1, bias=False), # Bx64x64x64 -> Bx128x64x64
			nn.BatchNorm2d(128),
			nn.ELU(),
			nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx128x64x64 -> Bx128x65x65
			nn.Conv2d(128, 128, 3, 2, 0, bias=False), #  Bx128x65x65 -> Bx128x32x32
			nn.BatchNorm2d(128),
			nn.ELU(),
			nn.Conv2d(128, 96, 3, 1, 1, bias=False), #  Bx128x32x32 -> Bx96x32x32
			nn.BatchNorm2d(96),
			nn.ELU(),
			nn.Conv2d(96, 192, 3, 1, 1, bias=False), #  Bx96x32x32 -> Bx192x32x32
			nn.BatchNorm2d(192),
			nn.ELU(),
			nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx192x32x32 -> Bx192x33x33
			nn.Conv2d(192, 192, 3, 2, 0, bias=False), # Bx192x33x33 -> Bx192x16x16
			nn.BatchNorm2d(192),
			nn.ELU(),
			nn.Conv2d(192, 128, 3, 1, 1, bias=False), # Bx192x16x16 -> Bx128x16x16
			nn.BatchNorm2d(128),
			nn.ELU(),
			nn.Conv2d(128, 256, 3, 1, 1, bias=False), # Bx128x16x16 -> Bx256x16x16
			nn.BatchNorm2d(256),
			nn.ELU(),
			nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx256x16x16 -> Bx256x17x17
			nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x17x17 -> Bx256x8x8
			nn.BatchNorm2d(256),
			nn.ELU(),
			nn.Conv2d(256, 160, 3, 1, 1, bias=False), # Bx256x8x8 -> Bx160x8x8
			nn.BatchNorm2d(160),
			nn.ELU(),
			nn.Conv2d(160, 320, 3, 1, 1, bias=False), # Bx160x8x8 -> Bx320x8x8
			nn.BatchNorm2d(320),
			nn.ELU(),
			nn.AvgPool2d(6, stride=1), #  Bx320x8x8 -> Bx320x1x1

		]
		self.G_enc_convLayers = nn.Sequential(*G_enc_convLayers)

		#Decoder
		G_dec_convLayers = [
			nn.ConvTranspose2d(320,160, 3,1,1, bias=False), # Bx320x8x8 -> Bx160x8x8
			nn.BatchNorm2d(160),
			nn.ELU(),
			nn.ConvTranspose2d(160, 256, 3,1,1, bias=False), # Bx160x8x8 -> Bx256x8x8
			nn.BatchNorm2d(256),
			nn.ELU(),
			nn.ConvTranspose2d(256, 256, 3,2,0, bias=False), # Bx256x8x8 -> Bx256x17x17
			nn.BatchNorm2d(256),
			nn.ELU(),
			Crop([0, 1, 0, 1]),
			nn.ConvTranspose2d(256, 128, 3,1,1, bias=False), # Bx256x16x16 -> Bx128x16x16
			nn.BatchNorm2d(128),
			nn.ELU(),
			nn.ConvTranspose2d(128, 192,  3,1,1, bias=False), # Bx128x16x16 -> Bx192x16x16
			nn.BatchNorm2d(192),
			nn.ELU(),
			nn.ConvTranspose2d(192, 192,  3,2,0, bias=False), # Bx128x16x16 -> Bx192x33x33
			nn.BatchNorm2d(192),
			nn.ELU(),
			Crop([0, 1, 0, 1]),
			nn.ConvTranspose2d(192, 96,  3,1,1, bias=False), # Bx192x32x32 -> Bx96x32x32
			nn.BatchNorm2d(96),
			nn.ELU(),
			nn.ConvTranspose2d(96, 128,  3,1,1, bias=False), # Bx96x32x32 -> Bx128x32x32
			nn.BatchNorm2d(128),
			nn.ELU(),
			nn.ConvTranspose2d(128, 128,  3,2,0, bias=False), # Bx128x32x32 -> Bx128x65x65
			nn.BatchNorm2d(128),
			nn.ELU(),
			Crop([0, 1, 0, 1]),
			nn.ConvTranspose2d(128, 64,  3,1,1, bias=False), # Bx128x64x64 -> Bx64x64x64
			nn.BatchNorm2d(64),
			nn.ELU(),
			nn.ConvTranspose2d(64, 64,  3,1,1, bias=False), # Bx64x64x64 -> Bx64x64x64
			nn.BatchNorm2d(64),
			nn.ELU(),
			nn.ConvTranspose2d(64, 64,  3,2,0, bias=False), # Bx64x64x64 -> Bx64x129x129
			nn.BatchNorm2d(64),
			nn.ELU(),
			Crop([0, 1, 0, 1]),
			nn.ConvTranspose2d(64, 32,  3,1,1, bias=False), # Bx64x128x128 -> Bx32x128x128
			nn.BatchNorm2d(32),
			nn.ELU(),
			nn.ConvTranspose2d(32, out_channel_num,  3,1,1, bias=False), # Bx32x128x128 -> Bxchx128x128
			nn.Tanh(),
		]
		self.G_dec_convLayers = nn.Sequential(*G_dec_convLayers)

		self.G_dec_fc = nn.Linear(320 + 50 + 10, 320*6*6)


	def forward(self, input_img, src_heatmap, dst_heatmap, desired_cond, fixed_noise):
		src_heatmap = src_heatmap.view(-1,1,96,96)
		input = torch.cat([input_img, src_heatmap], dim=1)

		x = self.G_enc_convLayers(input)

		x = x.view(-1,320)
		desired_cond = desired_cond.view(-1, 10)

		self.features = x

		x = torch.cat([desired_cond, x, fixed_noise], 1)

		x = self.G_dec_fc(x)

		x = x.view(-1, 320, 6, 6)

		x = self.G_dec_convLayers(x)

		return x, self.features