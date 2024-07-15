import os
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import torchvision
import numpy as np
from PIL import Image
import random
import cv2

import sys
sys.path.append('D:/1. Research/Paper/1. LDR loss code/LDRNET_240612/code/cython')
import geth2d

random.seed(123)

class trainLoader(data.Dataset):
	def __init__(self, imagePath):
		self.imagePath = imagePath
		self.inFiles = self.listFiles(os.path.join(imagePath, 'input'))	

	def getH2d(self, srcDATA):
		
		R, C = srcDATA.size
		src = srcDATA.load()

		h2D_in = np.zeros((256, 256), dtype=np.float32)
		for r in range(0, R):
			for c in range(0, C):
				ref = src[r, c]

				if r < R - 1:
					trg = src[r+1, c]
					h2D_in[max(trg, ref), min(trg, ref)] = h2D_in[max(trg, ref), min(trg, ref)] + 1
				if c < C - 1:
					trg = src[r, c+1]
					h2D_in[max(trg, ref), min(trg, ref)] = h2D_in[max(trg, ref), min(trg, ref)] + 1

		hvec = np.zeros((32640,), dtype=np.float32)
		maskLayer = np.zeros((255, ), dtype=np.float32)
		maskvec = np.zeros((32640, ), dtype=np.float32)

		vecidx = 0
		for layer in range(1, 256):
			hl = np.zeros((256-layer, ), dtype=np.float32)

			tmp_idx = 0
			for j in range(layer, 256):
				i = j - layer
				hl[tmp_idx] = np.log(h2D_in[j, i]+1)
				tmp_idx += 1
			
			sumh = np.sum(hl) # reliability of dl is proportional to hl
			if sumh == 0:
				vecidx = vecidx + 256-layer
				continue

			maskLayer[layer-1] = sumh
			maskvec[vecidx:vecidx + 256-layer] = sumh
			hvec[vecidx:vecidx + 256-layer] = hl/sumh #  kappa = 1 / sum(hl) by eq. (3) and (11)
			vecidx = vecidx + 256-layer
			
		maskLayer = maskLayer / np.max(maskLayer) # by eq. (23)
		maskvec = maskvec / np.max(maskvec)

		return hvec, maskLayer, maskvec
	
	def transform(self, image, GT):
		i, j, h, w = torchvision.transforms.RandomCrop.get_params(image, output_size=(256, 256))
		image = TF.crop(image, i, j, h, w)
		GT = TF.crop(GT, i, j, h, w)

		return image, GT


	def __getitem__(self, index):
		fname = os.path.split(self.inFiles[index])[-1]
		dataLow  = Image.open(self.inFiles[index])
		dataGT   = Image.open(os.path.join(self.imagePath, 'gt', fname))

		cropLow, cropGT = self.transform(dataLow, dataGT)

		cropL = cropLow.convert('L')
		#cropL = cropLow.convert('HSV')[2]

		dataH2d, dataMask, dataMaskVec = geth2d.getH2d(np.array(cropL))

		cropLow = (np.asarray(cropLow)/255.0)
		cropGT = (np.asarray(cropGT)/255.0)

		cropLow = torch.from_numpy(cropLow).float()
		cropGT = torch.from_numpy(cropGT).float()
		dataH2d = torch.from_numpy(dataH2d).float()
		dataMask = torch.from_numpy(dataMask).float()
		dataMaskVec = torch.from_numpy(dataMaskVec).float()
		
		cropLow = 2.0 * cropLow - 1.0
		cropGT = 2.0 * cropGT - 1.0

		return cropLow.permute(2,0,1), cropGT.permute(2,0,1), dataH2d, dataMask, dataMaskVec

	def __len__(self):
		return len(self.inFiles)

	def listFiles(self, inPath):
		files = []
		for (dirPath, dirNames, fileNames) in os.walk(inPath):
			files.extend(fileNames)
			break
		files = sorted([os.path.join(inPath, x) for x in files])
		return files

class testLoader(data.Dataset):
	def __init__(self, imagePath):
		self.imagePath = imagePath
		self.inFiles = self.listFiles(os.path.join(imagePath, 'input'))

	def __getitem__(self, index):
		fileName = os.path.split(self.inFiles[index])[-1]
		dataLow  = Image.open(self.inFiles[index])
		dataGT   = Image.open(os.path.join(self.imagePath, 'gt', fileName))

		dataLow = torchvision.transforms.Resize(size=(256,256))(dataLow)
		dataGT = torchvision.transforms.Resize(size=(256,256))(dataGT)

		dataInput = (np.asarray(dataLow)/255.0)
		dataGT = (np.asarray(dataGT)/255.0)

		dataInput = torch.from_numpy(dataInput).float()
		dataGT = torch.from_numpy(dataGT).float()
		
		dataInput = 2.0 * dataInput - 1.0
		dataGT = 2.0 * dataGT - 1.0

		return dataInput.permute(2,0,1), dataGT.permute(2,0,1)

	def __len__(self):
		return len(self.inFiles)

	def listFiles(self, inPath):
		files = []
		for (dirPath, dirNames, fileNames) in os.walk(inPath):
			files.extend(fileNames)
			break
		files = sorted([os.path.join(inPath, x) for x in files])
		return files