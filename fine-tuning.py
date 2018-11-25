import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import numpy as np

from model import resnet50

import os

from tensorboardX import SummaryWriter


class Deblur(nn.Module):
	
	def __init__(self, PRETRAINED):
		super(Deblur, self).__init__()
		self.exec = resnet50(pretrained=PRETRAINED)
		self.classifier = nn.Linear(2048, 2)
	
	def forward(self, x):
		x, _ = self.exec(x)
		cls_prob = self.classifier(x)
		return cls_prob


def train(ROOT, MODEL, LOSS, OPTIM):
	
	train_path = os.path.join(ROOT, 'fine_tuning_trainset')
	writer = SummaryWriter(os.path.join(os.path.expanduser('~'), 'codes', 'curriculum', 'tensorboard'))
	
	trans = transforms.Compose([
		transforms.Resize((224,224)),
		#transforms.RandomCrop((224,224)),
		#transforms.RandomVerticalFlip(p=0.5),
		transforms.ToTensor(),
		transforms.Normalize([0.485,0.456,0.406],
		                     [0.229,0.224,0.225])])

	
	trainset = ImageFolder(root=train_path, transform=trans)
	loader = torch.utils.data.DataLoader(trainset,
	                                     batch_size=4,
	                                     shuffle=True)
	
	for idx, data in enumerate(loader):
		
		prob = MODEL(data[0])
		import ipdb
		ipdb.set_trace()
		loss = LOSS(prob, data[1])
		

		OPTIM.zero_grad()
		loss.backward()
		OPTIM.step()
		
		# import ipdb
		# ipdb.set_trace()
		# writer.add_scalar('loss','{:.4f}'.format(loss.item()), idx)
		
		# print('loss','{:.4f}'.format(loss.item()), idx)
		# print('=====================>')
		print('iteration {}, loss {}'.format(idx, loss.item()))
		
	
	

if __name__ == '__main__':

	# models = ['resnet101', 'resnet50']
	# datasets = ['ICDAR15', 'MSRA-TD500', 'MSRA-TD500.blur']

	root = os.path.join(os.path.expanduser('~'), 'Documents', 'MSRA-TD500')
	
	model = Deblur(PRETRAINED=True)
	loss = nn.CrossEntropyLoss(size_average=True)
	optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

	# writer = SummaryWriter(os.path.join(os.path.expanduser('~'), 'codes', 'curriculum', 'tensorboard'))
	train(ROOT=root, MODEL=model, LOSS=loss, OPTIM=optimizer)