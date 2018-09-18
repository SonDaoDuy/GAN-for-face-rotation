import numpy as np
import os
from tqdm import tqdm
import argparse
import glob
import re
import pickle
import struct as st

parser = argparse.ArgumentParser()
parser.add_argument('-ii', '--input_images_dir', type=str, help='dir with imgs')
parser.add_argument('-ia', '--input_lmk_file', type=str, help='lmk file')
parser.add_argument('-op', '--output_path', type=str, help='Output path')
args = parser.parse_args()

def get_data(image_filepaths, lmk_file):
	#read data from bin file here
	f = open(lmk_file, 'rb')
	num_of_images, points_detected = st.unpack('ii', f.read(8))
	print("num_of_imgs: %d, points_detected: %d" % (num_of_images, points_detected))
	count_invalid = []
	for i in range(num_of_images):
		valid_bb = st.unpack('b',f.read(1))
		if valid_bb == 0:
			count_invalid.append(i)

	print("Number of invalid bb: %d" % (len(count_invalid)))

	#save data into data dict {'image_name': lmk_points}
	data = dict()
	print("number of image: %d" %(len(image_filepaths)))

	if len(image_filepaths) != num_of_images:
		print("Error: different size of img and lmk points!!")
		return None

	for filepath in tqdm(image_filepaths):
		points_lmk = st.unpack('d'*points_detected*2, f.read(8*points_detected*2))
		data[os.path.basename(filepath[:-4])] = np.array(points_lmk)

	return data

def save_dict(data, name):
	with open(name + '.pkl', 'wb') as f:
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def main():
	image_filepaths = glob.glob(os.path.join(args.input_images_dir, '*.jpg'))
	#co can sort ko????

	lmk_file = args.input_lmk_file

	#create lmk file
	data = get_data(image_filepaths, lmk_file)

	if data is not None:
		if not os.path.isdir(args.output_path):
			os.makedirs(args.output_path)

		save_dict(data, os.path.join(args.output_path, 'lmk_points'))
	else:
		print("Error in size of imgs and lmk point!!")

if __name__ == '__main__':
	main()