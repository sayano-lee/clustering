import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import json

from PIL import Image
import cv2

from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans

from skimage.feature import hog
from skimage import exposure

from utils import clustering, TrainSet


def main(model, dataset):
	
	if len(dataset.split('.')) == 1:
		train_root = os.path.join(os.path.expanduser('~'), 'Documents', dataset, 'trainim')
	else:
		dataset = dataset.split('.')[0]
		train_root = os.path.join(os.path.expanduser('~'), 'Documents', dataset, 'blur')
	
	root = '/home/litianjiao/codes/curriculum'
	
	# trans = transforms.Compose([
	# 	transforms.Resize((224, 224)),
	# 	transforms.ToTensor(),
	# 	transforms.Normalize([0.485, 0.456, 0.406],
	# 						[0.229, 0.224, 0.225])])
	
	trans = transforms.Resize((224, 224))
	
	# trainset = TrainSet(root=train_root, transform=trans)
	trainset_via_cv2 = TrainSet(root=train_root, loader='opencv.color', transform=trans)
	
	pbar = tqdm(total=len(trainset_via_cv2))
	
	index = []
	
	hog_means = []
	hog_vars = []
	
	for cnt, (im, id) in enumerate(trainset_via_cv2):

		index.append(id)
		
		
		fd, hog_image = hog(im, orientations=8, pixels_per_cell=(16, 16),
		                    cells_per_block=(1, 1), visualize=True, multichannel=True)
		
		hog_means.append(fd.mean())
		hog_vars.append(fd.var())

		# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
		
		# ax1.axis('off')
		# ax1.imshow(im, cmap=plt.cm.gray)
		# ax1.set_title('Input image')
		
		# Rescale histogram for better display
		# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
		
		# ax2.axis('off')
		# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
		# ax2.set_title('Histogram of Oriented Gradients')
		# plt.show()

		pbar.update(1)

	pbar.close()
		
	
	index_path = os.path.join(root, 'extracted_feats', model + '_' + dataset + '.index')


	with open(index_path, 'w') as f:
		json.dump(index, f)
		
	return hog_means, hog_vars
	
def nonDeepClustering(N_CLUSTERING, MEAN, VAR):


	y = KMeans(n_clusters=N_CLUSTERING).fit_predict(list(zip(MEAN, VAR)))
	color_map = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'wo']
	
	
	class_means = []
	class_vars = []

	for i in range(N_CLUSTERING):
		tmp_means = []
		tmp_vars = []
		for count, idx in enumerate(y):
			if idx == i:
				tmp_means.append(MEAN[count])
				tmp_vars.append(VAR[count])
		class_means.append(tmp_means)
		class_vars.append(tmp_vars)
		
	for j in range(N_CLUSTERING):
		plt.plot(class_means[j], class_vars[j], color_map[j])
	
	plt.savefig('NON_DEEP_mean_std_cls' + str(N_CLUSTERING))
	
	return y

if __name__ == '__main__':

	models = ['resnet101', 'resnet50', 'resnet18', 'vgg11', 'sobel', 'laplacian', 'HOG']
	datasets = ['ICDAR15', 'MSRA-TD500', 'MSRA-TD500.blur']
	num_of_clustering = 2
	
	dataset = datasets[1]
	model = models[6]

	image_root_path = os.path.join(os.path.expanduser('~'), 'Documents', dataset, 'trainim')
	root = '/home/litianjiao/codes/curriculum/extracted_feats'
	
	means, vars = main(model=model, dataset=dataset)
	y = nonDeepClustering(N_CLUSTERING=num_of_clustering, MEAN=means, VAR=vars)
	clustering(NUM_OF_CLUSTERING=num_of_clustering, IMG_ROOT=image_root_path, ROOT=root, ID=y,
	           DATASET=dataset, MODEL=model)
