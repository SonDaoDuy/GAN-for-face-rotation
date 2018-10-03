import csv
import os
import glob

def main():
	train_file = './sample_dataset_96/train_ids.csv'
	test_file = './sample_dataset_96/test_ids.csv'

	src_path = 'D:\\Face Normalize\\GAN_face\\sample_dataset_96\\imgs\\'

	img_paths = glob.glob(os.path.join(src_path, '*.jpg'), recursive=True)

	f_train = open(train_file, 'w')
	f_test = open(test_file, 'w')
	writer_train = csv.writer(f_train)
	writer_test = csv.writer(f_test)
	#print(len(img_path))
	count = 0
	for idx, img_path in enumerate(img_paths, 1):
		# img_name = img_path.split('\\')[-1]
		# img_id = img_name.split('.')[0]
		# img_name_save = img_id + '.jpg'
		img_name_save = os.path.basename(img_path)
		f_train.write(img_name_save)
		f_train.write('\n')
		f_test.write(img_name_save)
		f_test.write('\n')
		count += 1
		if count == 10000:
			break
		# if idx <= len(img_paths)*0.9:
		# 	f_train.write(img_name)
		# 	f_train.write('\n')
		# 	#writer_train.writerow([img_name])
		# else:
		# 	f_test.write(img_name)
		# 	f_test.write('\n')
		# 	#writer_test.writerow([img_name])

	f_train.close()
	f_test.close()

if __name__ == '__main__':
 	main() 
