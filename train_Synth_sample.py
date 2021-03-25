import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import scipy.io as scio
import argparse
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import re

from math import exp 
from data_loader_sample import SampleDataset , ToTensor
from Gaussian import main 
from data_utils import generate_gt

from collections import OrderedDict

from PIL import Image
from torchvision.transforms import transforms
from craft import CRAFT
from torch.autograd import Variable
from multiprocessing import Pool

import random

random.seed(10)

if __name__ == '__main__': 

    sample_dataset = SampleDataset(image_folder='/root/data/SynthText' ,
                                    imnames= '/root/data/SynthText/imnames_sample.npy',
                                    charBB = '/root/data/SynthText/charBB_sample.npy',
                                    aff_charBB='/root/data/SynthText/aff_charBB_sample.npy',
                                    transform=ToTensor())
    sample_train_loader = torch.utils.data.DataLoader(
        sample_dataset,
        batch_size = 16,
        shuffle = True, 
        num_workers = 0,
        drop_last = True,
        pin_memory = True)  #이게 뭐지??

    net = CRAFT()

    net = net.cuda()

    net = torch.nn.DataParallel(net, device_ids = [0,1]).cuda()
    cudnn.benchmark = True  #이게 뭐지? 

    optimizer = optim.Adam(net.parameters(), lr = 1e-4 )
    criterion = nn.MSELoss()

    net.train()

    loss_time = 0 
    loss_value = 0
    compare_loss = 1

    for epoch in range(10):
        loss_value = 0 

        st = time.time()
        for index, (image , region_label , affinity_label) in enumerate(sample_train_loader):
        
            image = Variable(image.type(torch.FloatTensor)).cuda()
            region_label = region_label.type(torch.FloatTensor)
            affinity_label = affinity_label.type(torch.FloatTensor)
            region_label = Variable(region_label).cuda()
            affinity_label = Variable(affinity_label).cuda()

            out, _ = net(image)

            optimizer.zero_grad()

            out1 = out[:, :, :, 0].cuda()
            out2 = out[:, :, :, 1].cuda()
            loss_region = criterion(out1 , region_label)
            loss_affinity = criterion(out2, affinity_label)

            loss = loss_region + loss_affinity

            loss.backward()
            optimizer.step()

            loss_value += loss.item()

            if index % 2 == 0 and index > 0:
                et = time.time()
                print('epoch {}:({}/{}) batch || training time for 16 batch {} || training loss {} ||'.format(epoch, index, len(sample_train_loader), et-st, loss_value/2))
                loss_time = 0
                loss_value = 0
                st = time.time()

            





