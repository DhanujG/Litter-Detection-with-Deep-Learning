
import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon


from taco_loader import *

from random import seed
from random import randint
import skimage.io
import skimage.transform



def load_modified_images():
    # Load class map - these tables map the original TACO classes to your desired class system
    # and allow you to discard classes that you don't want to include.
    class_map = {}
    with open("./data/all_image_urls.csv") as csvfile:
        reader = csv.reader(csvfile)
        class_map = {row[0]:row[1] for row in reader}

    # Load full dataset or a subset
    TACO_DIR = "../data"
    round = None # Split number: If None, loads full dataset else if int > 0 selects split no 
    subset = "train" # Used only when round !=None, Options: ('train','val','test') to select respective subset
    dataset = dataset.Taco()
    taco = dataset.load_taco(TACO_DIR, round, subset, class_map=class_map, return_taco=True)

    # Must call before using the dataset
    dataset.prepare()

    print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    bbox_widths = []
    bbox_heights = []
    obj_areas_sqrt = []
    obj_areas_sqrt_fraction = []
    bbox_aspect_ratio = []
    max_image_dim = 1024

    for ann in taco.dataset['annotations']:
    
        imgs = taco.loadImgs(ann['image_id'])
        
        resize_scale = max_image_dim/max(imgs[0]['width'], imgs[0]['height'])
        # Uncomment this to work on original image size
        #     resize_scale = 1
        
        bbox_widths.append(ann['bbox'][2]*resize_scale)
        bbox_heights.append(ann['bbox'][3]*resize_scale)
        obj_area = ann['bbox'][2]*ann['bbox'][3]*resize_scale**2 # ann['area']
        obj_areas_sqrt.append(np.sqrt(obj_area))
            
        img_area = imgs[0]['width']*imgs[0]['height']*resize_scale**2
        obj_areas_sqrt_fraction.append(np.sqrt(obj_area/img_area))

    print('According to MS COCO Evaluation. This dataset has:')
    print(np.sum(np.array(obj_areas_sqrt)<32), 'small objects (area<32*32 px)')
    print(np.sum(np.array(obj_areas_sqrt)<64), 'medium objects (area<96*96 px)')
    print(np.sum(np.array(obj_areas_sqrt)<96), 'large objects (area>96*96 px)')

    trash_images = []

    for index, image in enumerate(dataset.image_ids):
        # Load random image
        image_id = index
        image_ori = dataset.load_image(image_id)
        #masks_ori, _ = dataset.load_mask(image_id)

        image_dtype = image_ori.dtype
        #nr_annotations = np.shape(masks_ori)[-1]

        bboxes = utils.extract_bboxes(masks_ori)

        for bbox_id in range(len(bboxes)):
    
            image = image_ori
            masks = masks_ori
            
            bboxes_cpy = bboxes
            y1, x1, y2, x2 = bboxes_cpy[bbox_id]
            h, w = image.shape[:2]
            
            bbox_width = x2-x1
            bbox_height = y2-y1
            
            img_max_dim = max(h,w)
            bbox_max_dim = max(bbox_width,bbox_height)
            
            print('Image original shape:',image.shape)
            print('Bbox original shape:',y1, x1, y2, x2)
            
            bbox_max_dim_threshold_4_scaling = config.IMAGE_MAX_DIM*0.8


         #If bbox is big enough or too big, downsize full image
        if bbox_max_dim > bbox_max_dim_threshold_4_scaling:
            
            # Rescale
            downscale_at_least = min(1.,config.IMAGE_MAX_DIM/bbox_max_dim)
            downscale_min = config.IMAGE_MAX_DIM/img_max_dim
            
            scale = random.random()*(downscale_at_least-downscale_min)+downscale_min
            
            print('Downscaling image by',scale)
            print("Scale interval:", downscale_min, downscale_at_least)
            
            # Actually scale Image   
            image = skimage.transform.resize(image, (np.round(h * scale), np.round(w * scale)),order=1,
                                            mode="constant", preserve_range=True)
        
            h, w = image.shape[:2]
            img_max_dim = max(h,w)
        
            # Padding
            top_pad = (img_max_dim - h) // 2
            bottom_pad = img_max_dim - h - top_pad
            left_pad = (img_max_dim - w) // 2
            right_pad = img_max_dim - w - left_pad
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0).astype(image_dtype)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
            
            # Adjust mask and other vars
            #masks = utils.resize_mask(masks, scale, padding)
            #bboxes_cpy = utils.extract_bboxes(masks)
            y1, x1, y2, x2 = bboxes[bbox_id]
            h, w = image.shape[:2]
            
            print('Image resized shape:',image.shape)
        
        # Select crop around target annotation
        x0_min = max(x2-config.IMAGE_MAX_DIM,0) 
        x0_max = min(x1,w-config.IMAGE_MAX_DIM)
        x0 = randint(x0_min, x0_max)
        y0_min = max(y2-config.IMAGE_MAX_DIM,0) 
        y0_max = min(y1,h-config.IMAGE_MAX_DIM)
        y0 = randint(y0_min, y0_max)
        
        if padding:
            max_dim = config.IMAGE_MAX_DIM
            window = (max(top_pad,y0)-y0, max(left_pad,x0)-x0, min(window[2],y0+max_dim)-y0, min(window[3],x0+max_dim)-x0)
            print(window)
        
        # Crop
        crop = (y0, x0, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)
        image = image[y0:y0 + config.IMAGE_MAX_DIM, x0:x0 + config.IMAGE_MAX_DIM]
        masks = masks[y0:y0 + config.IMAGE_MAX_DIM, x0:x0 + config.IMAGE_MAX_DIM]
        ax[i+1].imshow(image)
        ax[i+1].axis('off')
        i+=1


        trash_images.append(image)

    return trash_images


#test data loader

def main():

    trash_images = load_modified_images()
    print(trash_images)


if __name__ == "__main__":
    main()
