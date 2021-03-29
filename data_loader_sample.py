from __future__ import print_function, division
import os
import cv2
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from scipy import io as scio
from Gaussian import main 
from data_utils import generate_gt

# 경고 메시지 무시하기
import warnings
warnings.filterwarnings("ignore")

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

class normalize(object):

    def __call__(self, image, region_score, affinity_score, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):

        img = image.copy().astype(np.float32)
        img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)

        return img , region_score, affinity_score

class ToTensor(object): 

    def __call__(self, image, region_score, affinity_score): 

        # numpy image : H x W x C 
        # torch image : C x H x C 
        image = image.transpose(2,0,1)

        return torch.from_numpy(image), torch.from_numpy(region_score) , torch.from_numpy(affinity_score)

class RandomCrop(object): 

    def __call__(self, image, region_score, affinity_score): 

        rand1 = random.random()*0.2 
        rand2 = random.random()*0.2 

        left = int(image.shape[0] * rand1)
        right = int(image.shape[0] - image.shape[0]* rand2)

        bottom = int(image.shape[1]*rand2)
        top = int(image.shape[1] - image.shape[1]*rand2)

        image = image[left:right, bottom:top, :]
        region_score = region_score[left:right , bottom:top , : ]
        affinity_score[left:right, bottom:top, :]

        return image, region_score, affinity_score

#Sample Data Set 
class SampleDataset(Dataset): 

    def __init__(self, image_folder, imnames, charBB , aff_charBB,transform=None ,target_size=768):

        self.image_folder = image_folder
        self.target_size = target_size
        self.heatmap = main(img_size=512, thred=3.5, test=False)
        self.imnames = np.load(imnames, allow_pickle=True)
        self.charBB = np.load(charBB, allow_pickle=True)
        self.aff_charBB = np.load(aff_charBB, allow_pickle=True)
        self.transform = transform

    def __len__(self): 
        return self.imnames.shape[0]

    def resize_gt(self, gt):
        return cv2.resize(gt, (self.target_size//2, self.target_size//2), interpolation = cv2.INTER_LINEAR)

    def resize(self, img, target_size):
        return cv2.resize(img, (self.target_size, self.target_size), interpolation = cv2.INTER_LINEAR)

    def __getitem__(self, index): 

        img_name = os.path.join( self.image_folder ,self.imnames[index][0]) 

        image = io.imread(img_name)
        region_score = generate_gt(img = image, heatmap=self.heatmap, bbox_cor=self.charBB[index].transpose(2,1,0))
        affinity_score = generate_gt(img = image, heatmap=self.heatmap, bbox_cor=self.aff_charBB[index])

        if self.transform: 
            image, region_score, affinity_score = self.transform(image, region_score, affinity_score)

        image = self.resize(img = image, target_size = self.target_size)
        region_score = self.resize_gt(gt = region_score)
        affinity_score = self.resize_gt(gt = affinity_score)

        

        return image, region_score, affinity_score

if __name__ == '__main__': 

    print('start test')

    sample_dataset = SampleDataset(image_folder='/root/data/SynthText' ,
                                    imnames= '/root/data/SynthText/imnames_sample.npy',
                                    charBB = '/root/data/SynthText/charBB_sample.npy',
                                    aff_charBB='/root/data/SynthText/aff_charBB_sample.npy',
                                    transform= transforms.Compose([ToTensor()]))

    img, region_score, affinity_score = sample_dataset[0]
    print('img shape : {}'.format(img.size()))
    print('region score shape : {}'.format(region_score.size()))
    print('affinity score shape : {}'.format(affinity_score.size()))