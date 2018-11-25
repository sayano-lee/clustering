import numpy as np
import os
import cv2
from tqdm import tqdm

def main(ROOT, SAVING_ROOT):
	
	files = os.listdir(ROOT)
	if not os.path.exists(SAVING_ROOT):
		os.makedirs(SAVING_ROOT)
		
	length = len(files)
	pbar = tqdm(total=length//2)
	
	for i in range(length//2):
		img = cv2.imread(os.path.join(ROOT, files[i]))

		'''
		generating the kernel
		'''
		size = 15
		kernel_motion_blur = np.zeros((size, size))
		kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
		kernel_motion_blur = kernel_motion_blur / size

		'''
		applying the kernel to the input image
		'''
		output = cv2.filter2D(img, -1, kernel_motion_blur)
		img_saving = os.path.join(SAVING_ROOT, files[i])
		cv2.imwrite(img_saving, output)
		
		pbar.update(1)
	
		# cv2.imshow(file, output)
		# cv2.waitKey(0)
	
	pbar.close()


if __name__ == '__main__':
	root = os.path.join(os.path.expanduser('~'), 'Documents', 'MSRA-TD500', 'trainim')
	save_root = os.path.join(os.path.expanduser('~'), 'Documents', 'MSRA-TD500', 'blur')
	main(ROOT=root, SAVING_ROOT=save_root)
