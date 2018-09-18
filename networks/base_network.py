import torch.nn as nn
import functools

class NetworkBase(nn.Module):
	"""docstring for NetworkBase"""
	def __init__(self):
		super(NetworkBase, self).__init__()
		self._name = 'BaseNetwork'

	@property
	def name(self):
		return self._name

	def init_weights(self):
		self.apply(self._weights_init_fn)


	#co can thiet them init weight cho linear (fc) layer
	def _weights_init_fn(self, m):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
			m.weight.data.normal_(0.0, 0.02)
			if hasattr(m.bias, 'data'):
				m.bias.data.fill_(0)
		elif classname.find('BatchNorm2d') != -1:
			m.weight.data.normal_(1.0, 0.02)
			m.bias.data.fill_(0)
		elif classname.find('Linear') != -1:
			m.weight.data.normal_(0, 0.02)
		elif classname.find('ConvTranspose2d') != -1:
			m.weight.data.normal_(0, 0.02)

	def _get_norm_layer(self, norm_type='batch'):
		if norm_type == 'batch':
			norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
		elif norm_type == 'instance':
			norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
		elif norm_type =='batchnorm2d':
			norm_layer = nn.BatchNorm2d
		else:
			raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

		return norm_layer

class Crop(nn.Module):
	"""
	Generator でのアップサンプリング時に， ダウンサンプル時のZeroPad2d と逆の事をするための関数
	論文著者が Tensorflow で padding='SAME' オプションで自動的にパディングしているのを
	ダウンサンプル時にはZeroPad2dで，アップサンプリング時には Crop で実現

	### init
	crop_list : データの上下左右をそれぞれどれくらい削るか指定
	"""

	def __init__(self, crop_list):
		super(Crop, self).__init__()

		# crop_lsit = [crop_top, crop_bottom, crop_left, crop_right]
		self.crop_list = crop_list

	def forward(self, x):
		B,C,H,W = x.size()
		x = x[:,:, self.crop_list[0] : H - self.crop_list[1] , self.crop_list[2] : W - self.crop_list[3]]

		return x