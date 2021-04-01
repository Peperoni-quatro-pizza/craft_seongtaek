import os 
import sys
import time 

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2 
import numpy as np 
from skimage import io 
import craft_utils
import imgproc
import json 
import file_utils
from data_utils import Gray2RGB

from collections import OrderedDict

from craft import CRAFT

weight_folder = '/root/data/model_param_sample'

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def test(image, epoch, index, cvt = False): 

    image = image

    print('input image shape {}'.format(image.shape))

    checkpoint = torch.load('/root/data/model_param_sample2/{}_{}.pth'.format(epoch, index))

    net = CRAFT().cuda()

    net.load_state_dict(copyStateDict(checkpoint['model_state_dict']))

    #이미지 리사이징 등등 

    #했다고 치고 진행 

    x = torch.from_numpy(image).permute(2,0,1)
    x = Variable(x.unsqueeze(0).type(torch.FloatTensor))
    x = x.cuda() 

    print(x.size())

    with torch.no_grad():
        y, _ = net(x)

    pred_region = y[0, : , : , 0].cpu().data.numpy()
    pred_affinity = y[0, : , :, 1].cpu().data.numpy()

    print(type(pred_region))
    print(pred_region.shape)


    # cvt == True -> Region, Affinity score H x W x C 
    # cvt == False -> Region, Affinity score H x W 
    if cvt: 
        pred_region = Gray2RGB(pred_region)
        pred_affinity = Gray2RGB(pred_affinity)

    return pred_region, pred_affinity

if __name__ == '__main__': 

    image = '/root/data/sample/sample_eng3.png'
    image_ = io.imread(image)

    print(image_.shape)

    #이미지가 alpha 채널이 있는경우 
    if image_.shape[2] == 4: 
        image_ = cv2.cvtColor(image_, cv2.COLOR_RGBA2RGB)

    region, affinity = test(image=image_, epoch=0, index=25000 , cvt=True)

    print('region_shape : {}'.format(region.shape))

    #!!!!!! 
    #Warning!! 
    #이미지 파일은 H x W  x C 이지만 cv2.resize 에서 사이즈는 W , H 순으로 넣는다 아래 코드 참고 

    region = cv2.resize(region, (image_.shape[1], image_.shape[0] ) ,cv2.INTER_LINEAR)
    affinity = cv2.resize(affinity, (image_.shape[1], image_.shape[0] ) ,cv2.INTER_LINEAR)


    io.imsave('/root/craft_re/sample_image/test/region_eng.jpg', region)
    io.imsave('/root/craft_re/sample_image/test/affinity_eng.jpg', affinity)

    print('image_shape[0] : {} , image_shape[1] : {}'.format(image_.shape[0], image_.shape[1]))

    add_weighted_image = cv2.addWeighted(image_ , 0.5 , region, 0.5 , 0 )

    io.imsave('/root/craft_re/sample_image/test/pred_eng.jpg', add_weighted_image)



