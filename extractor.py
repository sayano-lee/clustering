import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import resnet50, resnet18
from vgg import vgg11

import os
import json

from PIL import Image
import cv2

from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans

from utils import clustering, TrainSet



class Extractor(nn.Module):
	
	def __init__(self):
		super(Extractor, self).__init__()
		
		'''ResNet'''
		# self.exec = resnet50(pretrained=True)
		self.exec = resnet18(pretrained=True)

		
		'''VGG'''
		# vgg = vgg11(pretrained=True)
		# self.exec = nn.Sequential(*list(vgg.children())[:-1])
	
	def forward(self, x):
		x, f = self.exec(x)
		# import ipdb
		# ipdb.set_trace()
		# feat = self.exec(x)
		
		# feat = F.max_pool2d(feat, kernel_size=7)
		# feat = F.avg_pool2d(feat, kernel_size=7)

		# return feat
		return x


'''
class TrainSet(torch.utils.data.dataset.Dataset):
	def __init__(self, root, loader='PIL', transform=None):
		self.root = root
		self.ims = self.__filter(root)
		self.transform = transform
		self.loader = loader
	
		if loader == 'PIL':
			im_loader = self.__default_loader
		elif loader == 'opencv':
			im_loader = self.__cv_loader
		else:
			raise NotImplementedError

	def __getitem__(self, index, RESIZE=(224, 224)):
		## FIXME modified for one-file features
		image_path = os.path.join(self.root, self.ims[index])
		
		if self.loader == 'PIL':
			img = self.__default_loader(image_path)
			if self.transform is not None:
				return self.transform(img), self.ims[index]
			else:
				return img, self.ims[index]
		elif self.loader == 'opencv':
			# resize for resnet or vgg
			img = self.__cv_loader(image_path, resize=RESIZE)
			return img, self.ims[index]
		else:
			raise NotImplementedError

	def __len__(self):
		return len(self.ims)

	@classmethod
	def __filter(cls, root):
		img_files = []
		files = os.listdir(root)
		for file in files:
			if os.path.basename(file).split('.')[1].lower() == 'jpg':
				img_files.append(file)
				
		return img_files

	@classmethod
	def __default_loader(cls, x):
		return Image.open(x).convert("RGB")
	
	@classmethod
	def __cv_loader(cls, x, resize):
		if not isinstance(resize, tuple):
			resize = (resize, resize)
		im = cv2.imread(x, 0)
		return cv2.resize(im, resize)
'''


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
	trainset_via_cv2 = TrainSet(root=train_root, loader='opencv', transform=trans)
	
	pbar = tqdm(total=len(trainset_via_cv2))
	
	ext = Extractor()
	for params in ext.parameters():
		params.requires_grad = False
	
	feats = []
	index = []
	
	lap_means = []
	lap_vars = []
	
	for cnt, (im, id) in enumerate(trainset_via_cv2):
		# import ipdb
		# ipdb.set_trace()

		## FIXME: uncomment for deep method
		# feat = ext(im.unsqueeze(0))
		# feats.append(feat)
		index.append(id)
		
		sobel_x = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=5)
		sobel_y = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=5)
		laplacian = cv2.Laplacian(im, cv2.CV_64F)
		
		laplacian_mean = laplacian.mean() / im.size
		laplacian_var = laplacian.var() / im.size
		
		lap_means.append(laplacian_mean)
		lap_vars.append(laplacian_var)
		
		
		pbar.update(1)

	pbar.close()
	
	## FIXME: uncomment for deep method
	# feats = torch.cat(feats).detach().numpy()
	# saving_path = os.path.join(root, 'extracted_feats', model + '_' + dataset + '.npy')
	# index_path = os.path.join(root, 'extracted_feats', model + '_' + dataset + '.index')
	# np.save(saving_path, feats)
	# with open(index_path, 'w') as f:
	# 	json.dump(index, f)
	
	index_path = os.path.join(root, 'extracted_feats', 'NON_DEEP_' + dataset + '.index')
	with open(index_path, 'w') as f:
		json.dump(index, f)
		
	return lap_means, lap_vars
	
def nonDeepClustering(N_CLUSTERING, MEAN, VAR):


	y = KMeans(n_clusters=N_CLUSTERING).fit_predict(list(zip(MEAN, VAR)))
	color_map = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'wo']
	
	
	class_means = []
	class_vars = []
	clustering_ids = []

	for i in range(N_CLUSTERING):
		tmp_means = []
		tmp_vars = []
		# tmp_ids = []
		for count, idx in enumerate(y):
			if idx == i:
				tmp_means.append(MEAN[count])
				tmp_vars.append(VAR[count])
		class_means.append(tmp_means)
		class_vars.append(tmp_vars)
		
	
	# import ipdb
	# ipdb.set_trace()

	for j in range(N_CLUSTERING):
		plt.plot(class_means[j], class_vars[j], color_map[j])

	
	plt.savefig('NON_DEEP_mean_std_cls' + str(N_CLUSTERING))

	return y

if __name__ == '__main__':

	models = ['resnet101', 'resnet50', 'resnet18', 'vgg11', 'sobel', 'laplacian']
	datasets = ['ICDAR15', 'MSRA-TD500', 'MSRA-TD500.blur']
	num_of_clustering = 2
	
	
	dataset = datasets[1]


	image_root_path = os.path.join(os.path.expanduser('~'), 'Documents', dataset, 'trainim')
	root = '/home/litianjiao/codes/curriculum/extracted_feats'
	
	# main(model=models[4], dataset=datasets[0])
	means, vars = main(model=models[4], dataset=dataset)
	y = nonDeepClustering(N_CLUSTERING=num_of_clustering, MEAN=means, VAR=vars)
	clustering(NUM_OF_CLUSTERING=num_of_clustering, IMG_ROOT=image_root_path, ROOT=root, ID=y)
