import scipy.io as scio
import numpy as np 
from skimage import io 
import cv2 
import matplotlib.pyplot as plt 
from skimage import color 
import os 
import re

from torch import dtype 

from data_utils import * 
from Gaussian import main 
from data_utils import generate_gt

image_folder='/root/data/SynthText'
imnames= '/root/data/SynthText/imnames.npy'
charBB = '/root/data/SynthText/charBB.npy'
aff_charBB='/root/data/SynthText/aff_charBB.npy'

if __name__ == '__main__':

    heatmap = main(img_size=512, thred=3.5, test=False)
    imnames = np.load(imnames, allow_pickle=True)
    charBB = np.load(charBB, allow_pickle=True)
    aff_charBB = np.load(aff_charBB, allow_pickle=True)

    num_image = imnames.shape[0]
    print('Save region, affinity score , number of images : {}'.format(num_image))

    for save_num in range(int(num_image/100)): 

        save_region = np.empty(100, dtype=object)
        save_affinity = np.empty(100, dtype=object)

        for index in range(100): 

            img_name = os.path.join(image_folder, imnames[save_num*100 + index][0])
            image = io.imread(img_name)
            save_region[index] = generate_gt(img= image, heatmap = heatmap, bbox_cor=charBB[save_num*100 + index].transpose(2,1,0))
            save_affinity[index] = generate_gt(img = image, heatmap= heatmap, bbox_cor=aff_charBB[save_num*100 + index])

        np.save('/root/data/SynthText/region/{}'.format(save_num) , save_region)
        np.save('/root/data/SynthText/affinity/{}'.format(save_num), save_affinity)

        print('Now {} / {}'.format(save_num*100+index , num_image))






         

