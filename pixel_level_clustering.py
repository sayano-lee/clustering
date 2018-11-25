import torchvision.transforms as transforms
import PIL
import numpy as np
import matplotlib.pyplot as plt

from extractor import TrainSet

from sklearn.cluster import KMeans

import os
import shutil


from tqdm import tqdm



def main(NUM_OF_CLUSTERING, ROOT):

	trainset = TrainSet(root=os.path.join(ROOT, 'trainim'),
	                    transform=transforms.Compose(
		                    [transforms.Resize((224,224))]
	                    ))
	
	means = []
	stds = []
	ids = []
	for cnt, (im, id) in enumerate(trainset):
		grey = np.sum(np.array(im), axis=2)//3
		means.append(np.mean(grey))
		stds.append(np.std(grey))
		ids.append(id)
	
	class_means = []
	class_stds = []
	cluster_ids = []
	
	color_map = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'wo']
	
	y = KMeans(n_clusters=NUM_OF_CLUSTERING).fit_predict(list(zip(means, stds)))
	
	for i in range(NUM_OF_CLUSTERING):
		tmp_means = []
		tmp_stds = []
		tmp_ids = []
		for count, idx in enumerate(y):
			if idx == i:
				tmp_means.append(means[count])
				tmp_stds.append(stds[count])
				tmp_ids.append(ids[count])

		class_means.append(tmp_means)
		class_stds.append(tmp_stds)
		cluster_ids.append(tmp_ids)
		
	

	for j in range(NUM_OF_CLUSTERING):
		plt.plot(class_means[j], class_stds[j], color_map[j])
	
	return cluster_ids


def clustering(NUM_OF_CLUSTERING, ROOT, ID):

	pbar = tqdm(total=len(os.listdir(os.path.join(ROOT, 'trainim'))))
	cluster_path = os.path.join(ROOT, 'clustering')
	if not os.listdir(cluster_path):
		for k in range(NUM_OF_CLUSTERING):
			os.mkdir(os.path.join(cluster_path, str(k)))
	else:
		shutil.rmtree(cluster_path)
	
	for idx, ids in enumerate(ID):
		cpath = os.path.join(cluster_path, str(idx))
		for id in ids:
			src = os.path.join(ROOT, 'trainim', id + '.jpg')
			shutil.copy(src, cpath)
			pbar.update(1)
	pbar.close()
	

if __name__ == '__main__':
	train_root = '/home/litianjiao/Documents/ICDAR15/end2end/'
	# train_root = '/home/litianjiao/Documents/MSRA-TD500/'
	root = '/home/litianjiao/codes/curriculum'
	
	num_of_clustering = 7
	
	ids = main(NUM_OF_CLUSTERING=num_of_clustering, ROOT=train_root)
	clustering(NUM_OF_CLUSTERING=num_of_clustering, ROOT=train_root, ID=ids)
