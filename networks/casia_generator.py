import torch.nn as nn
import numpy as np
from .base_network import NetworkBase, Crop
import torch

class CasiaGenerator(NetworkBase):
	"""

	"""
	def __init__(self, in_channel_num=4, out_channel_num=3):
		super(CasiaGenerator, self).__init__()
		self._name = 'casia_generator'
		self.features = []

		#Encoder
		G_enc_convLayers = [
			nn.Conv2d(in_channel_num, 32, 3, 1, 1, bias=False), # Bx3x128x128 -> Bx32x128x128
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
			nn.AvgPool2d(8, stride=1), #  Bx320x8x8 -> Bx320x1x1

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
			#nn.Tanh(),
		]
		self.G_dec_convLayers = nn.Sequential(*G_dec_convLayers)

		self.G_dec_fc = nn.Linear(320, 320*8*8)

		#weight initialize
		# for m in self.modules():
		# 	if isinstance(m, nn.Conv2d):
		# 		m.weight.data.normal_(0, 0.02)

		# 	elif isinstance(m, nn.ConvTranspose2d):
		# 		m.weight.data.normal_(0, 0.02)

		# 	elif isinstance(m, nn.Linear):
		# 		m.weight.data.normal_(0, 0.02)

	def forward(self, input_img, input_heatmap):
		input_heatmap = input_heatmap.view(-1,1,128,128)
		input = torch.cat([input_img, input_heatmap], dim=1)

		x = self.G_enc_convLayers(input)

		x = x.view(-1,320)

		self.features = x

		x = self.G_dec_fc(x)

		x = x.view(-1, 320, 8, 8)

		x = self.G_dec_convLayers(x)

		return x, self.features

# class Crop(nn.Module):
# 	"""
# 	Generator でのアップサンプリング時に， ダウンサンプル時のZeroPad2d と逆の事をするための関数
# 	論文著者が Tensorflow で padding='SAME' オプションで自動的にパディングしているのを
# 	ダウンサンプル時にはZeroPad2dで，アップサンプリング時には Crop で実現

# 	### init
# 	crop_list : データの上下左右をそれぞれどれくらい削るか指定
# 	"""

# 	def __init__(self, crop_list):
# 		super(Crop, self).__init__()

# 		# crop_lsit = [crop_top, crop_bottom, crop_left, crop_right]
# 		self.crop_list = crop_list

# 	def forward(self, x):
# 		B,C,H,W = x.size()
# 		x = x[:,:, self.crop_list[0] : H - self.crop_list[1] , self.crop_list[2] : W - self.crop_list[3]]

# 		return x