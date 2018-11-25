import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import os
import json

from tqdm import tqdm
import shutil

def main(NUM_OF_CLUSTERING, ROOT, MODEL, DATASET):
	
	
	path = os.path.join(ROOT, MODEL + '_' + DATASET + '.npy')
	with open(path, 'rb') as f:
		feats = np.load(f)
	
	
	means = np.mean(feats,axis=1)
	stds = np.std(feats, axis=1)
	
	import ipdb
	ipdb.set_trace()
	
	class_means = []
	class_stds = []
	cluster_ids = []
	
	# blue green red cyan magenta yellow black white
	color_map = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'wo']
	
	#y = KMeans(n_clusters=NUM_OF_CLUSTERING).fit_predict(list(zip(means, stds)))
	y = KMeans(n_clusters=NUM_OF_CLUSTERING).fit_predict(list(zip(stds, means)))
	
	for i in range(NUM_OF_CLUSTERING):
		tmp_means = []
		tmp_stds = []
		for count, idx in enumerate(y):
			if idx == i:
				tmp_means.append(means[count])
				tmp_stds.append(stds[count])
		
		class_means.append(tmp_means)
		class_stds.append(tmp_stds)
	
	for j in range(NUM_OF_CLUSTERING):
		#plt.plot(class_means[j], class_stds[j], color_map[j])
		plt.plot(class_stds[j], class_means[j], color_map[j])

	plt.savefig(MODEL + '_' + DATASET + '_mean_std' + '_cls' + str(NUM_OF_CLUSTERING))
	
	return y


def clustering(NUM_OF_CLUSTERING, IMG_ROOT, ROOT, ID, MODEL, DATASET):
	
	pbar = tqdm(total=len(os.listdir(IMG_ROOT)))
	cluster_path = os.path.join(ROOT, MODEL + '_' + DATASET + '_clustering')
	index_path = os.path.join(ROOT, MODEL + '_' + DATASET + '.index')
	
	with open(index_path, 'r') as f:
		index = json.load(f)
	
	if os.path.exists(cluster_path):
		shutil.rmtree(cluster_path)
		os.makedirs(cluster_path)
	else:
		os.makedirs(cluster_path)
		
	for k in range(NUM_OF_CLUSTERING):
		os.mkdir(os.path.join(cluster_path, str(k)))
	
	for id, idx in enumerate(ID):
		cpath = os.path.join(cluster_path, str(idx))
		
		src = os.path.join(IMG_ROOT, index[id])
		shutil.copy(src, cpath)
		pbar.update(1)
		
	pbar.close()

if __name__ == '__main__':

	root = '/home/litianjiao/codes/curriculum/extracted_feats'

	num_of_clustering = 3

	models = ['resnet101', 'resnet50', 'resnet18', 'vgg11']
	datasets = ['ICDAR15', 'MSRA-TD500']
	
	model = models[2]
	dataset = datasets[0]
	image_root_path = os.path.join(os.path.expanduser('~'), 'Documents', dataset, 'trainim')
	
	y = main(NUM_OF_CLUSTERING=num_of_clustering, ROOT=root, MODEL=model, DATASET=dataset)
	clustering(NUM_OF_CLUSTERING=num_of_clustering, IMG_ROOT=image_root_path, ROOT=root, ID=y, \
	           MODEL=model, DATASET=dataset)
