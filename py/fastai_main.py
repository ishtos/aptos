import os
import sys
import gc
import time
import random
import cv2
import glob
import requests
import json
import math
import re
import hashlib
import psutil

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from functools import partial
from collections import Counter
from PIL import Image
from multiprocessing import cpu_count
from tqdm import tqdm_notebook as tqdm
from multiprocessing import Pool
from statistics import median
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from fastai import *
from fastai.vision import *
from fastai.callbacks import *

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

import albumentations as A
from albumentations import torch as AT


from efficientnet_pytorch import EfficientNet

def getImageMetaData(strFile):
    file = None;
    bRet = False;
    strMd5 = "";
    
    try:
        file = open(strFile, "rb");
        md5 = hashlib.md5();
        strRead = "";
        
        while True:
            strRead = file.read(8096);
            if not strRead:
                break;
            md5.update(strRead);
        #read file finish
        bRet = True;
        strMd5 = md5.hexdigest();
    except:
        bRet = False;
    finally:
        if file:
            file.close()

    return strMd5

def crop_image_from_gray(image, tol=8):
    if image.ndim == 2:
        mask = image>told
        return image[np.ix_(mask.any(1),mask.any(0))]
    elif image.ndim== 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = gray_image>tol        
        check_shape = image[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return image
        else:
            image1=image[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            image2=image[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            image3=image[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            image = np.stack([image1,image2,image3],axis=-1)
        return image

def _load_format(path, convert_mode, after_open)->Image:
    image_size = 300
    image = cv2.imread(path)
    image = crop_image_from_gray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, depth = image.shape
    rate = height / width
    height = int(image_size * rate)
    width = image_size
    if height < 300:
        width = width * (300 / height)
        height = 300

    image = cv2.resize(image, (width, height))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image[:, :, 0] = clahe.apply(image[:, :, 0]) 
    image[:, :, 1] = clahe.apply(image[:, :, 1]) 
    image[:, :, 2] = clahe.apply(image[:, :, 2]) 
    
    x = width // 2
    y = height // 2
    r = np.amin((x, y))

    circle_image = np.zeros((height, width), np.uint8)
    cv2.circle(circle_image, (x, y), int(r), 1, thickness=-1)
    image = cv2.bitwise_and(image, image, mask=circle_image)
    image = crop_image_from_gray(image)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image , (0,0) , 30) , -4, 128) 

    return Image(pil2tensor(image, np.float32).div_(255))

def get_df():
    base_image_dir = os.path.join('..', 'input', 'aptos2019-blindness-detection')
    train_dir = os.path.join(base_image_dir, 'train_images')
    df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))
    df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))
    image_meta_l = Parallel(n_jobs=psutil.cpu_count(), verbose=1)((delayed(getImageMetaData)(fp) for fp in df['path']))
    df['strMd5'] = image_meta_l
    df['strMd5_count'] = df.groupby('strMd5')['id_code'].transform('count')
    df = df.drop_duplicates(subset=['diagnosis', 'strMd5'])
    df = df.drop(columns=['id_code', 'strMd5', 'strMd5_count'], axis=1)
    df = df.sample(frac=1).reset_index(drop=True) 
    test_df = pd.read_csv(os.path.join(base_image_dir, 'sample_submission.csv'))

    _, valid, _, _ = train_test_split(df, df['diagnosis'], test_size=0.2, random_state=43)
    
    return df, test_df, valid.index

def get_old_df():
    base_image_dir = os.path.join('..', 'input', 'diabetic-retinopathy-resized-png')
    train_dir = os.path.join(base_image_dir, 'resized_train_cropped', 'resized_train_cropped')
    df = pd.read_csv(os.path.join('..', 'py', 'resample_previous.csv'))
    df['path'] = df['image'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))
    df = df.drop(columns=['image'])
    df = df.sample(frac=1).reset_index(drop=True) 
    df = df.rename(columns={'level': 'diagnosis'})
    idx_s = df.shape[0]

    train_df, _, _ = get_df()
    df = pd.concat([df, train_df], axis=0).reset_index()

    idx_e = df.shape[0]

    return df, idx_s, idx_e


def qk(y_pred, y):
    return torch.tensor(cohen_kappa_score(torch.round(y_pred), y, weights='quadratic'), device='cuda:0')


def seed_everything():
    seed = 43
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    vision.data.open_image = _load_format
    seed_everything()

    df, test_df, valid_idx = get_df()
    old_df, idx_s, idx_e = get_old_df()

    print("START LAOD")
    
    model = EfficientNet.from_pretrained('efficientnet-b3')
    model._fc = nn.Linear(in_features=model._fc.in_features, out_features=1, bias=True)

    print("END LOAD")

    bs = 32
    size = 300

    tfms = ([
        RandTransform(tfm=TfmAffine (dihedral_affine), kwargs={}, p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True),
        RandTransform(tfm=TfmAffine (zoom), kwargs={'scale': (1.0, 1.3)}, p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True),
        RandTransform(tfm=TfmAffine (rotate), kwargs={'degrees': (-180.0, 180.0)}, p=0.75, resolved={}, do_run=True, is_random=True, use_on_y=True),
        RandTransform(tfm=TfmLighting (contrast), kwargs={'scale': (0.7, 1.45)}, p=0.75, resolved={}, do_run=True, is_random=True, use_on_y=True),
    ],
    [
        RandTransform(tfm=TfmAffine (zoom), kwargs={'scale': (1.0, 1.3)}, p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True),
    ])

    old_data = (ImageList.from_df(df=old_df, path='./', cols='path')
        .split_by_idx(list(range(idx_s, idx_e))) 
        .label_from_df(cols='diagnosis', label_cls=FloatList)
        .transform(tfms=tfms, size=size, resize_method=ResizeMethod.SQUISH, padding_mode='zeros') 
        .databunch(bs=bs, num_workers=0) 
        .normalize(imagenet_stats)  
       )

    learn = Learner(data=old_data, 
                model=model, 
                path='../',
                model_dir='weights',
                metrics=[qk]).to_fp16()

    print("START OLD TRAIN")

    learn.freeze_to(20)
    learn.fit_one_cycle(5, 0.0005)
    learn.save(os.path.join('stage-1-epoch-5-model-3'))

    print("END OLD TRAIN")

    data = (ImageList.from_df(df=df, path='./', cols='path') 
        .split_by_idx(valid_idx)
        .label_from_df(cols='diagnosis', label_cls=FloatList)
        .transform(tfms=tfms, size=size, resize_method=ResizeMethod.SQUISH, padding_mode='zeros') 
        .databunch(bs=bs, num_workers=0) 
        .normalize(imagenet_stats)  
       )

    learn = Learner(data, 
                model,   
                path='../',
                model_dir='weights',
                metrics=[qk]).to_fp16()

    learn.data.add_test(ImageList.from_df(test_df,
                                      os.path.join('..', 'input', 'aptos2019-blindness-detection'),
                                      folder='test_images',
                                      suffix='.png'))

    print("START TRAIN")
   
    learn.unfreeze()
    learn.fit_one_cycle(10, 0.0005)
    learn.save(os.path.join('stage-2-epoch-15-model-3'))

    for i in range(0, 10):
        learn.fit_one_cycle(1, 0.0001)
        learn.save(os.path.join(f'stage-3-epoch-{i}-model-3'))

    print("END TRAIN")


if __name__ == '__main__':
    main()
