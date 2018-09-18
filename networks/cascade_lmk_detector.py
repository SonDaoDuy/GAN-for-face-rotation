import torch.nn as nn
import numpy as np
from .base_network import NetworkBase
import torch

class CascadeLmkDectector(NetworkBase):
	"""docstring for CascadeLmkDectector"""
	def __init__(self, in_channel_num=3):
		super(CascadeLmkDectector, self).__init__()
		self._name = 'cascade_lmk_detector'

		layers = []
		layers.append(nn.Conv2d(in_channel_num, 20, 4, 1, 0, bias=False)) # bn x in_chn x 128 x 128 -> bn x 20 x 125 x 125
		layers.append(nn.MaxPool2d((2,2)))                                # bn x 20 x 125 x 125 -> bn x 20 x 62 x 62
		layers.append(nn.Conv2d(20, 40, 3, 1, 0, bias=False))             # bn x 20 x 62 x 62 -> bn x 40 x 60 x 60
		layers.append(nn.MaxPool2d((2,2)))                                # bn x 40 x 60 x 60 -> bn x 40 x 30 x 30
		layers.append(nn.Conv2d(40, 60, 3, 1, 0, bias=False))             # bn x 40 x 30 x 30 -> bn x 60 x 28 x 28
		layers.append(nn.MaxPool2d((2,2)))                                # bn x 60 x 28 x 28 -> bn x 60 x 14 x 14
		layers.append(nn.Conv2d(60, 80, 2, 1, 0, bias=False))             # bn x 60 x 14 x 14 -> bn x 80 x 13 x 13

		self.conv = nn.Sequential(*layers)
		self.fc1 = nn.Linear(80*13*13, 120)
		self.fc2 = nn.Linear(120, 10)  

		# #weight initialize
		# for m in self.modules():
		# 	if isinstance(m, nn.Conv2d):
		# 		m.weight.data.normal_(0, 0.02)

		# 	elif isinstance(m, nn.ConvTranspose2d):
		# 		m.weight.data.normal_(0, 0.02)

		# 	elif isinstance(m, nn.Linear):
		# 		m.weight.data.normal_(0, 0.02)

	def forward(self, input):
		x = self.conv(input)

		x = x.view(-1, 80*13*13)

		x = self.fc1(x)

		x = x.view(-1, 120)

		x = self.fc2(x)

		return x 