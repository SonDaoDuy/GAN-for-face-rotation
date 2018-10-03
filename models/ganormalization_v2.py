import torch
from collections import OrderedDict
from torch.autograd import Variable
import utils.util as util
import utils.plots as plot_utils
from .base_model import BaseModel
from networks.network_factory import NetworksFactory
import os
import numpy as np
from utils.cv_utils import *

class GANormalizationv2(BaseModel):
	"""docstring for GANormalization"""
	def __init__(self, opt):
		super(GANormalizationv2, self).__init__(opt)
		self._name = 'ganormalization_v2'

		#create network
		self._init_create_network()

		#init train variables
		if self._is_train:
			self._init_train_vars()

		#load networks and optimizaers
		if not self._is_train or self._opt.load_epoch > 0:
			self.load()

		#prefetch variables
		self._init_prefetch_inputs()

		#init
		self._init_losses()

	def _init_create_network(self):
		#generator
		#self._G = self._create_generator()
		#self._G = self._create_generator_casia_v2() #change img size to 96x96, use res block (not latent vec)
		self._G = self._create_generator_casia_v3() # change model G (concat target pose at latent)
		#self._G = self._create_generator_wgan()
		#self._G = self._create_generator_unet()
		self._G.init_weights()
		if len(self._gpu_ids) > 1:
			self._G = torch.nn.DataParallel(self._G, device_ids=self._gpu_ids)
		self._G.cuda()

		#discriminator
		self._D = self._create_discriminator_wgan() 
		self._D.init_weights()
		if len(self._gpu_ids) > 1:
			self._D = torch.nn.DataParallel(self._D, device_ids=self._gpu_ids)
		self._D.cuda()
		#print(self._D)

		#light cnn for id preserve
		#self._D_id = self._create_light_cnn()
		self._D_id = self._create_openface_net() # change image size to 96x96
		#self._D_id = torch.nn.DataParallel(self._D_id).cuda()
		#model.load_state_dict(torch.load(os.path.join(containing_dir, 'openface.pth')))
		checkpoint = torch.load(self._opt.face_reg_cnn)
		self._D_id.load_state_dict(checkpoint)
		# for param in self._D_id.parameters():
		# 	param.requires_grad = False
		# if len(self._gpu_ids) > 1:
		# 	self._D_id = torch.nn.DataParallel(self._D_id, device_ids=self._gpu_ids)
		# self._D_id.cuda()

		print("Done create network!!")

	def _create_generator(self):
		return NetworksFactory.get_by_name('casia_generator')

	def _create_generator_casia_v2(self):
		return NetworksFactory.get_by_name('casia_generator_v2')

	def _create_generator_casia_v3(self):
		return NetworksFactory.get_by_name('casia_generator_v3')

	def _create_generator_wgan(self):
		return NetworksFactory.get_by_name('generator_wgan')

	def _create_generator_unet(self):
		return NetworksFactory.get_by_name('unet_generator')

	def _create_lmk_detector(self):
		return NetworksFactory.get_by_name('cascade_lmk_detector')

	def _create_lmk_detector_v2(self):
		return NetworksFactory.get_by_name('cascade_lmk_detector_v2')

	def _create_discriminator_wgan(self):
		return NetworksFactory.get_by_name('discriminator_wgan')

	def _create_light_cnn(self):
		return NetworksFactory.get_by_name('LightCNN_29v2')

	def _create_openface_net(self):
		return NetworksFactory.get_by_name('openface_net')

	def _init_train_vars(self):
		self._current_lr_G =  self._opt.lr_G
		#self._current_lr_D_lmk = self._opt.lr_D_lmk
		self._current_lr_D = self._opt.lr_D

		#initialize optimizers
		self._optimizer_G = torch.optim.Adam(self._G.parameters(), lr=self._current_lr_G,
											betas=[self._opt.G_adam_b1, self._opt.G_adam_b2])
		self._optimizer_D = torch.optim.Adam(self._D.parameters(), lr=self._current_lr_D,
											betas=[self._opt.D_adam_b1, self._opt.D_adam_b2])		
		# self._optimizer_D_lmk = torch.optim.Adam(self._D_lmk.parameters(), lr=self._current_lr_D_lmk,
		# 									betas=[self._opt.D_lmk_adam_b1, self._opt.D_lmk_adam_b2])

		print("Done create train vars!!")

	def _init_prefetch_inputs(self):
		self._input_real_img = self._Tensor(self._opt.batch_size, 3, self._opt.image_size, self._opt.image_size)
		self._clone_of_fake_img = self._Tensor(self._opt.batch_size, 3, self._opt.image_size, self._opt.image_size)
		self._input_real_cond = self._Tensor(self._opt.batch_size, self._opt.cond_nc)
		self._input_desired_cond = self._Tensor(self._opt.batch_size, self._opt.cond_nc)
		self._input_real_heat_map = self._Tensor(self._opt.batch_size, 1, self._opt.image_size, self._opt.image_size)
		self._input_desired_heat_map = self._Tensor(self._opt.batch_size, 1, self._opt.image_size, self._opt.image_size)
		self._input_real_img_path = None
		self._clone_feat = self._Tensor(self._opt.batch_size, 320)
		self._input_real_cond_path = None
		#self._input_desired_cond = None

		print("Done init inputs!!")

	def _init_losses(self):
		#define loss function
		self._criterion_cycle = torch.nn.L1Loss().cuda()
		self._criterion_D_cond = torch.nn.MSELoss().cuda()
		#self._criterion_D_cond = torch.nn.MSELoss().cuda()

		#init losses G
		self._loss_g_cond = Variable(self._Tensor([0]))
		self._loss_g_fake = Variable(self._Tensor([0]))
		self._loss_g_rec_real = Variable(self._Tensor([0]))

		
		#init losses D
		self._loss_d_cond = Variable(self._Tensor([0]))
		self._loss_d_real = Variable(self._Tensor([0]))
		self._loss_d_fake = Variable(self._Tensor([0]))
		self._loss_d_gp = Variable(self._Tensor([0]))

		print("Done init loss!!")


	def set_input(self, input):
		self._input_real_img.resize_(input['real_img'].size()).copy_(input['real_img'])
		self._input_real_cond.resize_(input['real_cond'].size()).copy_(input['real_cond'])
		self._input_desired_cond.resize_(input['desired_cond'].size()).copy_(input['desired_cond'])
		self._input_real_heat_map.resize_(input['real_heatmap'].size()).copy_(input['real_heatmap'])
		self._input_desired_heat_map.resize_(input['desired_heatmap'].size()).copy_(input['desired_heatmap'])
		self._input_real_id = input['sample_id']
		self._input_real_img_path = input['real_img_path']

		if len(self._gpu_ids) > 0:
			self._input_real_img = self._input_real_img.cuda(self._gpu_ids[0], async=True)
			self._input_real_cond = self._input_real_cond.cuda(self._gpu_ids[0], async=True)
			self._input_desired_cond = self._input_desired_cond.cuda(self._gpu_ids[0], async=True)

	def set_train(self):
		self._G.train()
		self._D.train()
		self._D_id.train()
		#self._D_lmk.train()
		self._is_train = True

	def set_eval(self):
		self._G.eval()
		self._is_train = False

	# get image paths
	def get_image_paths(self):
		return OrderedDict([('real_img', self._input_real_img_path)])

	def forward(self, keep_data_for_visuals=False, return_estimates=False):
		if not self._is_train:
			#convert tensor to variables
			real_img = Variable(self._input_real_img, volatile=True)
			real_cond = Variable(self._input_real_cond, volatile=True)
			desired_cond = Variable(self._input_desired_cond, volatile=True)
			real_heat_map = Variable(self._input_real_heat_map)
			desired_heat_map = Variable(self._input_desired_heat_map)

			#generate fake images
			# fake_img, _ = self._G.forward(real_img, desired_heat_map)
			# rec_real_img, _ = self._G.forward(fake_img, real_heat_map)

			fake_img, _ = self._G.forward(real_img, real_heat_map, desired_heat_map, desired_cond)
			rec_real_img, _ = self._G.forward(fake_img, desired_heat_map, real_heat_map, real_cond)

			imgs = None
			data = None
			if return_estimates:
				im_real_img = util.tensor2im(real_img.data)
				im_fake_img = util.tensor2im(fake_img.data)
				im_rec_real_img = util.tensor2im(rec_real_img.data)
				im_concat_img_batch = np.concatenate([im_real_img, im_fake_img, im_rec_real_img], 1)

				imgs = OrderedDict([('real_img', im_real_img),
									('fake_img', im_fake_img),
									('rec_real_img', im_rec_real_img),
									('concat', im_concat_img_batch)
									])
				data = OrderedDict([('real_path', self._input_real_img_path),
									('desired_cond', desired_cond.data[0,...].cpu().numpy().astype('str'))
									])
			if keep_data_for_visuals:
				self._vis_real_img = util.tensor2im(self._input_real_img)
				self._vis_fake_img= util.tensor2im(fake_img.data)
				self._vis_rec_real_img = util.tensor2im(rec_real_img.data)

				self._vis_real_cond = self._input_real_cond.cpu()[0, ...].numpy()
				self._vis_desired_cond = self._input_desired_cond.cpu()[0, ...].numpy()
				
				self._vis_batch_real_img = util.tensor2im(self._input_real_img, idx=-1)
				self._vis_batch_fake_img = util.tensor2im(fake_img.data, idx=-1)
				self._vis_batch_rec_real_img = util.tensor2im(rec_real_img.data, idx=-1) #!!!!!!

			return imgs, data


	def optimize_parameters(self, train_generator=True, keep_data_for_visuals=True):
		if self._is_train:
			#convert tensor to variables
			self._B = self._input_real_img.size(0)
			self._real_img = Variable(self._input_real_img)
			#print(self._real_img.size())
			self._real_cond = Variable(self._input_real_cond)
			self._desired_cond = Variable(self._input_desired_cond)
			self._desired_heat_map = Variable(self._input_desired_heat_map)
			self._real_heat_map = Variable(self._input_real_heat_map)
			self._fixed_noise = Variable(torch.FloatTensor(np.random.uniform(-1,1, (self._opt.batch_size, 50))).cuda())

			#train D
			loss_d, fake_img_pose = self._forward_D()
			self._optimizer_D.zero_grad()
			loss_d.backward()
			self._optimizer_D.step()


			loss_D_gp= self._gradinet_penalty_D(fake_img_pose)
			self._optimizer_D.zero_grad()
			loss_D_gp.backward()
			self._optimizer_D.step()

			#train G
			if train_generator:
				loss_g = self._forward_G(keep_data_for_visuals)
				self._optimizer_G.zero_grad()
				loss_g.backward()
				self._optimizer_G.step()
		
	def _forward_G(self, keep_data_for_visuals):
		#generate fake images
		fake_img, feat_real = self._G.forward(self._real_img, self._real_heat_map, self._desired_heat_map, self._desired_cond, self._fixed_noise)
		self._clone_feat.resize_(feat_real.size()).copy_(feat_real.data)
		clone_feat = Variable(self._clone_feat)

		#Discriminator
		d_fake_img_prob, d_fake_desire_cond, _ = self._D.forward(fake_img, self._desired_heat_map)
		self._loss_g_fake = self._compute_loss_D(d_fake_img_prob, True) * self._opt.lambda_D_prob
		self._loss_g_cond = self._criterion_D_cond(d_fake_desire_cond, self._desired_cond) / self._B * self._opt.lambda_D_cond

		#go to light cnn for id compare
		# input_1 = fake_img.sum(1)
		# input_1 = input_1.view(self._B, 1, 128, 128)

		# input_2 = self._real_img.sum(1)
		# input_2 = input_2.view(self._B, 1, 128, 128)

		# fake_prob, fake_feat = self._D_id.forward(input_1)
		# real_prob, real_feat = self._D_id.forward(input_2)

		#go to openface
		(fake_prob, fake_feat) = self._D_id.forward(fake_img)
		(real_prob, real_feat) = self._D_id.forward(self._real_img)

		self._loss_g_sim = (self._criterion_D_cond(fake_feat, real_feat.detach()) +\
			self._criterion_D_cond(fake_prob, real_prob.detach())) / self._B *self._opt.lambda_G_sim

		#cycle
		rec_real_img, feat_fake = self._G.forward(fake_img, self._desired_heat_map, self._real_heat_map, self._real_cond, self._fixed_noise)
		self._loss_g_rec_real = self._criterion_cycle(rec_real_img, self._real_img) * self._opt.lambda_cyc

		#self._loss_g_img_1_smooth = self._compute_loss_smooth(fake_img) * self._opt.lambda_mask_smooth
		#self._loss_g_img_2_smooth = self._compute_loss_smooth(rec_real_img) * self._opt.lambda_mask_smooth

		#self._loss_g_sim_2 = self._criterion_D_cond(feat_fake, clone_feat) / self._B *self._opt.lambda_G_sim
		# keep data for visualization
		if keep_data_for_visuals:
			self._vis_real_img = util.tensor2im(self._input_real_img)

			self._vis_fake_img= util.tensor2im(fake_img.data)

			self._vis_rec_real_img = util.tensor2im(rec_real_img.data)
			
			self._vis_batch_real_img = util.tensor2im(self._input_real_img, idx=-1)
			self._vis_batch_rec_real_img = util.tensor2im(rec_real_img.data, idx=-1)
			self._vis_batch_fake_img = util.tensor2im(fake_img.data, idx=-1)
			
		return self._loss_g_cond + self._loss_g_rec_real + self._loss_g_fake +\
				self._loss_g_sim #+ self._loss_g_sim_2

	def _forward_D(self):
		#generate fake images
		#print(self._real_img.size())
		#print(self._real_img.size())
		fake_img, _ = self._G.forward(self._real_img, self._real_heat_map, self._desired_heat_map, self._desired_cond, self._fixed_noise)
		#print(fake_img.size())
		#D(real_I)
		self._d_fake_real_img_prob, d_real_img_cond, _ = self._D.forward(self._real_img, self._real_heat_map)
		# print(d_fake_real_img.size())
		# print(self._real_img.size())
		self._loss_d_real = self._compute_loss_D(self._d_fake_real_img_prob, True) * self._opt.lambda_D_prob
		self._loss_d_cond = self._criterion_D_cond(d_real_img_cond, self._real_cond) / self._B * self._opt.lambda_D_cond
		#D(fake_I)
		# self._clone_of_fake_img.resize_(fake_img.size()).copy_(fake_img.data)
		# clone_image = Variable(self._clone_of_fake_img)
		self._d_fake_fake_img_prob, _, _ = self._D.forward(fake_img.detach(), self._desired_heat_map)
		self._loss_d_fake = self._compute_loss_D(self._d_fake_fake_img_prob, False) * self._opt.lambda_D_prob
		#self._loss_d_rec_fake = torch.mean(torch.abs(fake_img - d_fake_fake_img))
		
		#self._loss_d_adv = self._loss_d_rec_real - self._loss_d_rec_fake*self._opt.param_k

		return self._loss_d_real + self._loss_d_cond + self._loss_d_fake, fake_img
	
	def _gradinet_penalty_D(self, fake_imgs_masked):
		# interpolate sample (in our case we concat the heat map here?????)
		alpha = torch.rand(self._B, 1, 1, 1).cuda().expand_as(self._real_img)
		interpolated = Variable(alpha * self._real_img.data + (1 - alpha) * fake_imgs_masked.data, requires_grad=True)
		interpolated_prob, _, _ = self._D(interpolated, interpolated)

		# compute gradients
		grad = torch.autograd.grad(outputs=interpolated_prob,
									inputs=interpolated,
									grad_outputs=torch.ones(interpolated_prob.size()).cuda(),
									retain_graph=True,
									create_graph=True,
									only_inputs=True)[0]

		# penalize gradientss
		grad = grad.view(grad.size(0), -1)
		grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
		self._loss_d_gp = torch.mean((grad_l2norm - 1) ** 2) * self._opt.lambda_D_gp

		return self._loss_d_gp

	def _compute_loss_D(self, estim, is_real):
		return -torch.mean(estim) if is_real else torch.mean(estim)

	def _compute_loss_smooth(self, mat):
		return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
				torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

	def save(self, label):
		# save networks
		self._save_network(self._G, 'G', label)
		self._save_network(self._D, 'D', label)

		# save optimizers
		self._save_optimizer(self._optimizer_G, 'G', label)
		self._save_optimizer(self._optimizer_D, 'D', label)

	def load(self):
		load_epoch = self._opt.load_epoch

		# load G
		self._load_network(self._G, 'G', load_epoch)

		if self._is_train:
			# load D
			self._load_network(self._D, 'D', load_epoch)

			# load optimizers
			self._load_optimizer(self._optimizer_G, 'G', load_epoch)
			self._load_optimizer(self._optimizer_D, 'D', load_epoch)
	
	def get_current_visuals(self):
		# visuals return dictionary
		visuals = OrderedDict()

		# input visuals
		title_input_img = os.path.basename(self._input_real_img_path[0])
		#visuals['1_input_img'] = plot_utils.plot_lmk(self._vis_real_img, self._vis_real_cond, title=title_input_img)
		# visuals['2_fake_img'] = plot_utils.plot_lmk(self._vis_fake_img, self._vis_desired_cond)
		# visuals['3_rec_real_img'] = plot_utils.plot_lmk(self._vis_rec_real_img, self._vis_real_cond)
		visuals['1_input_img'] = self._vis_real_img
		visuals['2_fake_img'] = self._vis_fake_img
		visuals['3_rec_real_img'] = self._vis_rec_real_img
		visuals['4_batch_real_img'] = self._vis_batch_real_img
		visuals['5_batch_fake_img'] = self._vis_batch_fake_img
		visuals['6_batch_rec_real_img'] = self._vis_batch_rec_real_img
		

		return visuals

	def get_current_errors(self):
		loss_dict = OrderedDict([('g_cond', self._loss_g_cond.data[0]),
			('g_fake', self._loss_g_fake.data[0]),
			('g_rec_real', self._loss_g_rec_real.data[0]),
			# ('g_img_smooth_1', self._loss_g_img_1_smooth.data[0]),
			#('g_sim_2', self._loss_g_sim_2.data[0]),
			('g_sim', self._loss_g_sim.data[0]),
			('d_cond', self._loss_d_cond.data[0]),
			('d_real', self._loss_d_real.data[0]),
			('d_fake', self._loss_d_fake.data[0]),
			('d_gd', self._loss_d_gp.data[0])])
		

		return loss_dict

	def get_current_scalars(self):
		return OrderedDict([('lr_G', self._current_lr_G), ('lr_D', self._current_lr_D)])

	def update_learning_rate(self):
		# updated learning rate G
		lr_decay_G = self._opt.lr_G / self._opt.nepochs_decay
		self._current_lr_G -= lr_decay_G
		for param_group in self._optimizer_G.param_groups:
		    param_group['lr'] = self._current_lr_G
		print('update G learning rate: %f -> %f' %  (self._current_lr_G + lr_decay_G, self._current_lr_G))

		# update learning rate D
		lr_decay_D = self._opt.lr_D / self._opt.nepochs_decay
		self._current_lr_D -= lr_decay_D
		for param_group in self._optimizer_D.param_groups:
			param_group['lr'] = self._current_lr_D
		print('update D learning rate: %f -> %f' %  (self._current_lr_D + lr_decay_D, self._current_lr_D))

