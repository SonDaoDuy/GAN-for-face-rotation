import os.path
import torchvision.transforms as transforms
from data.dataset import DatasetBase
from PIL import Image
import random
import numpy as np
import pickle
from utils import cv_utils
import matplotlib.pyplot as plt

class CasiaDataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(CasiaDataset, self).__init__(opt, is_for_train)
        self._name = 'CasiaDataset'
        #self._count_dsi = 0

        #read dataset
        self._read_dataset_paths()

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        # start_time = time.time()
        real_img = None
        real_cond = None
        real_heat_map = None
        
        while real_img is None or real_cond is None or real_heat_map is None:
            # if sample randomly: overwrite index
            if not self._opt.serial_batches:
                index = random.randint(0, self._dataset_size - 1)

            # get sample data
            sample_id = self._ids[index]

            real_img, real_img_path = self._get_img_by_id(sample_id) #!!!!
            real_cond = self._get_cond_by_id(sample_id) # !!!!!
            #print(len(real_cond))
            real_heat_map =self._cond_to_heat_map(real_cond)
            real_heat_map = real_heat_map/real_heat_map.max()
            #print(np.shape(real_heat_map))

            if real_img is None:
                print ('error reading image %s, skipping sample' % sample_id)
            if real_cond is None:
                print ('error reading aus %s, skipping sample' % sample_id)
            if real_heat_map is None:
                print ('error making heatmap %s, skipping sample' % sample_id)

        desired_cond = None
        desired_heat_map = None
        while desired_heat_map is None or desired_cond is None:
            desired_cond = self._generate_random_cond()
            #print(len(desired_cond))
            #tao desired_heat_map o day
            desired_heat_map = self._cond_to_heat_map(desired_cond)
            desired_heat_map = desired_heat_map/desired_heat_map.max()
            #print("error here!!")

        # transform data
        img = self._transform(Image.fromarray(real_img))
        # desired_heat_map = transforms.ToTensor(desired_heat_map)
        # real_heat_map = transforms.ToTensor(real_heat_map)

        # pack data
        sample = {'real_img': img,
                  'real_cond': real_cond,
                  'desired_cond': desired_cond,
                  'real_heatmap': real_heat_map,
                  'desired_heatmap': desired_heat_map,
                  'sample_id': sample_id,
                  'real_img_path': real_img_path
                  }

        # print (time.time() - start_time)

        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset_paths(self):
        self._root = self._opt.data_dir
        self._imgs_dir = os.path.join(self._root, self._opt.images_folder)

        # read ids
        use_ids_filename = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
        use_ids_filepath = os.path.join(self._root, use_ids_filename)
        self._ids = self._read_ids(use_ids_filepath) #ham read nay

        # read lmk
        conds_filepath = os.path.join(self._root, self._opt.lmk_file)
        self._conds = self._read_conds(conds_filepath) #ham read nay

        self._ids = list(set(self._ids).intersection(set(self._conds.keys())))

        # dataset size
        self._dataset_size = len(self._ids)

    def _create_transform(self):
        if self._is_for_train:
            transform_list = [transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]),
                              ]
        else:
            transform_list = [transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]),
                              ]
        self._transform = transforms.Compose(transform_list)
        print("It goes here!!!")

    def _read_ids(self, file_path):
        ids = np.loadtxt(file_path, delimiter='\t', dtype=np.str)
        return [id[:-4] for id in ids]

    def _read_conds(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def _get_img_by_id(self, id):
        filepath = os.path.join(self._imgs_dir, id+'.jpg')
        #print(filepath)
        return cv_utils.read_cv2_img(filepath), filepath

    def _get_cond_by_id(self, id):
        if id in self._conds:
            return self._conds[id]
        else:
            return None

    def _generate_random_cond(self):
        index = random.randint(0, self._dataset_size - 1)
        id = self._ids[index]
        if id in self._conds:
            return self._conds[id]
        else:
            return None

    def _cond_to_heat_map(self,cond_lmk):
        x = []
        y = []
        if cond_lmk is None:
             print('error in making x, y heat map!!!, len x: %d, len y: %d' % (len(x), len(y)))
             return None

        for i in range(5):
            x_item = np.random.normal(cond_lmk[i*2], 2, 5000)
            y_item = np.random.normal(cond_lmk[i*2+1], 2, 5000)
            x.extend(x_item)
            y.extend(y_item)

        H, xedges, yedges = np.histogram2d(x, y, bins=96, range = ((0,96), (0,96)))
        #H, xedges, yedges = np.histogram2d(x, y, bins=(96, 96))
        H = H.T
        #heatmap = H[::-1]
        # print("done here")
        return H

    # def _test_heat_map(self, cond_lmk, image):
    #     x = []
    #     y = []
    #     if cond_lmk is None:
    #          print('error in making x, y heat map!!!, len x: %d, len y: %d' % (len(x), len(y)))
    #          return None

    #     for i in range(5):
    #         x_item = np.random.normal(cond_lmk[i*2], 2, 25000)
    #         y_item = np.random.normal(-cond_lmk[i*2+1], 2, 25000)
    #         x.extend(x_item)
    #         y.extend(y_item)

    #     H, xedges, yedges = np.histogram2d(x, y, bins=128, range = ((0,128), (-128,0)))
    #     #H, xedges, yedges = np.histogram2d(x, y, bins=(96, 96))
    #     H = H.T
    #     plt.figure(1)
    #     plt.subplot(221)
    #     plt.pcolormesh(x, y, H)
    #     plt.subplot(222)
    #     plt.imshow(image)
    #     plt.show()