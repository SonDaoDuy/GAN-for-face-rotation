from utils.face_utils import *
from utils.cv_utils import *
from utils.util import *
import glob
import os
def main():
	root_folder = 'C:\\Users\\duyson\\Desktop\\Projects\\FaceNormalize\\PytorchGAN\\dataset\\CASIA-WebFace\\CASIA-WebFace\\Small_Piece_For_Easy_Download\\CASIA-WebFace\\CASIA-WebFace\\'

	src_img_path = os.path.join(root_folder + '**\\*.jpg')
	print(src_img_path)
	src_img_folders = glob.glob(src_img_path, recursive=True)
	dst_img_folder = 'D:\\Face Normalize\\GAN_face\\sample_dataset\\imgs\\'
	count = 0
	print(len(src_img_folders))
	for img_path in src_img_folders:
		#thuc hien face detection va move sang folder sample_dataset\\imgs
		img_name = img_path.split('\\')[-1]
		img_folder = img_path.split('\\')[-2]
		img_data = read_cv2_img(img_path)
		
		if img_data is not None:
			bb = detect_biggest_face(img_data)
			if bb != None:
				img_to_save = crop_face_with_bb(img_data, bb)
				img_to_save = resize_face(img_to_save)
				name_to_save = img_folder + '_' + img_name
				save_path = os.path.join(dst_img_folder, name_to_save)
				save_image(img_to_save, save_path)
				count += 1
				print(count)
				# if count == 200000:
				# 	break

if __name__ == '__main__':
	main()


