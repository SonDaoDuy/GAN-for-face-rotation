import os
import argparse
import glob
import cv2
from utils import face_utils
from utils import cv_utils
import face_recognition
from PIL import Image
import torchvision.transforms as transforms
import torch
import pickle
import numpy as np
from models.models import ModelsFactory
from options.test_options import TestOptions

class MorphFacesInTheWild:
    def __init__(self, opt):
        self._opt = opt
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._model.set_eval()
        self._transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])
                                              ])

    def morph_file(self, img_path, desired_cond):
        img = cv_utils.read_cv2_img(img_path)
        morphed_img = self._morph_face(img, desired_cond) #!!!!!!
        output_name = '%s_out.png' % os.path.basename(img_path)
        self._save_img(morphed_img, output_name)# !!!!!!!

    def _morph_face(self, face, desired_cond):
        face = torch.unsqueeze(self._transform(Image.fromarray(face)), 0)
        expresion = torch.unsqueeze(torch.from_numpy(desired_cond), 0)
        real_heat_map = self._cond_to_heat_map(desired_cond)
        desired_heat_map = self._cond_to_heat_map(desired_cond)
        test_batch = {
        'real_img': face, 
        'real_cond': expresion, 
        'desired_cond': expresion, 
        'real_heatmap': real_heat_map,
        'desired_heatmap': desired_heat_map, 
        'sample_id': torch.FloatTensor(), 
        'real_img_path': []
        }
        self._model.set_input(test_batch)

        #forward model
        real_img = Variable(self._model._input_real_img, volatile=True)
        real_heat_map = Variable(self._model._input_real_heat_map, volatile=True)
        desired_heat_map = Variable(self._model._input_desired_heat_map, volatile=True)

        #generate fake images
        fake_img, _ = self._model._G.forward(real_img, desired_heat_map)
        #rec_real_img, _ = self._G.forward(fake_img, real_heat_map)
        im_concat_img = np.concatenate([real_img, fake_img],1)

        # im_real_img = util.tensor2im(real_img.data)
        # im_fake_img = util.tensor2im(fake_img.data)
        # im_rec_real_img = util.tensor2im(rec_real_img.data)
        #imgs, _ = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
        return im_concat_img

    def _save_img(self, img, filename):
        filepath = os.path.join(self._opt.output_dir, filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img)

    def _cond_to_heat_map(self,cond_lmk):
        x = []
        y = []
        if cond_lmk is None:
            print('error in making x, y heat map!!!, len x: %d, len y: %d' % (len(x), len(y)))
            return None

	    for i in range(5):
            x_item = np.random.normal(cond_lmk[i*2], 2, 50000)
            y_item = np.random.normal(cond_lmk[i*2+1], 2, 50000)
            x.extend(x_item)
            y.extend(y_item)

        H, xedges, yedges = np.histogram2d(x, y, bins=(128, 128))
        return H

def main():
    opt = TestOptions().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    morph = MorphFacesInTheWild(opt)

    image_path = opt.input_path
    # choose a pose and make heat map
     # read lmk
    conds_filepath = os.path.join(opt.data_dir, opt.lmk_file)
    _conds = _read_conds(conds_filepath) #ham read nay

    _ids = list(set(_ids).intersection(set(_conds.keys())))
    _dataset_size = len(_ids)

    #gen heat map
    for i in range(10):
        desired_cond = _generate_random_cond(_dataset_size, _ids, _conds)
        morph.morph_file(image_path, desired_cond)

def _read_conds(self, file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def _generate_random_cond(_dataset_size, _ids, _conds):
        index = random.randint(0, _dataset_size - 1)
        id = _ids[index]
        if id in _conds:
            return _conds[id]
        else:
            return None



if __name__ == '__main__':
    main()