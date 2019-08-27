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

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from functools import partial
from collections import Counter
from PIL import Image
from multiprocessing import cpu_count
from tqdm import tqdm
from multiprocessing import Pool

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

config = {
    # settings
    'seed': 43,
    'num_workers': cpu_count(),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # data
    'previous_data_path': os.path.join('..', 'input', 'diabetic-retinopathy-resized'),
    'data_path': os.path.join('..', 'input', 'aptos2019-blindness-detection'),
    
    # optimizer
    'optimizer_name': 'Adam',
    
    # model
    'model_name': 'efficientnet-b3',
    'pretrained': False,
#     'weight_path': os.path.join('..', 'input', 'resnet101', 'resnet101.pth'),
    # 'weight_path': os.path.join('..', 'input', 'efficientnet-aptos', 'efficientnet-b3.pth'),
    
    # loss
    'loss_name': 'MSE',
    
    # transforms
    'transforms': 'pytorch',
#     'transforms': 'albumentations',
    
    'pytorch': {
        'resize': {'train': False, 'test': False, 'train_size': 224, 'test_size': 224},
        'centerCrop': {'train': False, 'test': False, 'train_size': 224,'test_size': 224},
        'colorJitter': {'train': True, 'test': False, 'brightness': 0.2, 'contrast': 0, 'saturation': 0, 'hue': 0},
        'randomCrop': {'train': True, 'test': False, 'train_size': 300, 'test_size': 300, 'padding': 4},
        'randomResizedCrop': {'train': False, 'test': False, 'train_size': 224, 'test_size': 224},
        'randomHorizontalFlip': {'train': True, 'test': False, 'p': 0.5},
        'randomAffine': {'train': True, 'test': False, 'degrees': 360, 'translate': None, 'scale': (1, 1.5), 'shear': None},
        'randomRotation': {'train': False, 'test': False, 'degrees': 360},
        'toTensor': {'train': True, 'test': True},
        'normalize': {'train': True, 'test': True},
        'randomErasing' : {'train': False, 'test': False, 'p': 0.1, 'value': 'random'}
    },
    
    # 'albumentations': {
    #     'resize': {'train': True, 'test': True, 'train_size': 256, 'test_size': 256},
    #     'centerCrop': {'train': False, 'test': False, 'train_size': 224,'test_size': 224}, # not work
    #     'horizontalFlip': {'train': True, 'test': False},
    #     'rotate': {'train': True, 'test': False, 'limit': 180},
    #     'clahe': {'train': True, 'test': False},
    #     'gaussNoise': {'train': True, 'test': False},
    #     'randomBrightness': {'train': True, 'test': False},
    #     'randomContrast': {'train': True, 'test': False},
    #     'randomBrightnrssContrast': {'train': False, 'test': False},
    #     'hueSaturationValue': {'train': True, 'test': False},
    #     'toTensor': {'train': True, 'test': True},
    #     'normalize': {'train': True, 'test': True},
    # },
    
    # train settings
    'num_classes': 1,
    'lr': 0.0001,
    'epochs': 100,
    'image_size': 512,
    'patience': 10,
    'verbose': True,
    'batch_size': 32,
    'valid_size': 0.2,
}

def seed_everything():
    seed = config['seed']
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def crop_image_from_gray(image, tol=7):
    if image.ndim == 2:
        mask = image > tol
        return image[np.ix_(mask.any(1), mask.any(0))]
   
    elif image.ndim == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = gray_image > tol
        
        check_shape = image[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0): 
            return image
        
        else:
            imageR = image[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            imageG = image[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            imageB = image[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            image = np.stack([imageR, imageG, imageB], axis=-1)
        
        return image
    
def preprocess(image_name):
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (config['image_size'], config['image_size']))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 30), -4, 128)

    return image

def get_optimizer(params): 
    optimizer_name = config['optimizer_name']
    lr = config['lr']
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(params=params, lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(params=params, lr=lr)
    elif optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(params=params, lr=lr)
    elif optimizer_name == 'Adadelta':
        optimizer = optim.Adadelta(params=params, lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(params=params, lr=lr)
    
    return optimizer

class QK(nn.Module):
    def __init__(self):
        super(QK, self).__init__()
        self.loss = None
        
    def forward(self, output, target):
        self.loss = torch.tensor(cohen_kappa_score(torch.round(output), target, weights='quadratic'), requires_grad=True)
        return self.loss

def get_model():
    model_name = config['model_name']
    pretrained = config['pretrained']
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
    elif model_name == 'vgg11_bn':
        model = models.vgg11_bn(pretrained=pretrained)
    elif model_name == 'vgg13_bn':
        model = models.vgg13_bn(pretrained=pretrained)
    elif model_name == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=pretrained)
    elif model_name == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=pretrained)
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=pretrain)
    elif model_name == 'efficientnet-b0':
        model = EfficientNet.from_pretrained(model_name, num_classes=1)
    elif model_name == 'efficientnet-b1':
        model = EfficientNet.from_pretrained(model_name, num_classes=1)
    elif model_name == 'efficientnet-b2':
        model = EfficientNet.from_pretrained(model_name, num_classes=1)
    elif model_name == 'efficientnet-b3':
        model = EfficientNet.from_pretrained(model_name, num_classes=1)
    elif model_name == 'efficientnet-b4':
        model = EfficientNet.from_pretrained(model_name, num_classes=1)
    elif model_name == 'efficientnet-b5':
        model = EfficientNet.from_pretrained(model_name, num_classes=1)
        
    return model

def get_optimizer(params): 
    optimizer_name = config['optimizer_name']
    lr = config['lr']
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(params=params, lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(params=params, lr=lr)
    elif optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(params=params, lr=lr)
    elif optimizer_name == 'Adadelta':
        optimizer = optim.Adadelta(params=params, lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(params=params, lr=lr)
    
    return optimizer

class QK(nn.Module):
    def __init__(self):
        super(QK, self).__init__()
        self.loss = None
        
    def forward(self, output, target):
        self.loss = torch.tensor(cohen_kappa_score(torch.round(output), target, weights='quadratic'), requires_grad=True)
        return self.loss

def get_loss():
    loss_name = config['loss_name']
    if loss_name == 'MSE':
        loss = nn.MSELoss()
    elif loss_name == 'CrossEntropy':
        loss = nn.CrossEntropyLoss()
    elif loss_name == 'BCE':
        loss = nn.BCELoss()
    elif loss_name == 'BCEWithLogits':
        loss = nn.BCEWithLogitsLoss()
    elif loss_name == 'QK':
        loss = QK()
    
    return loss

def get_label_data(train):
    
    return train['diagnosis'].values
                                
def get_transforms_train():
    transforms_train_list = []

    if config['transforms'] == 'pytorch':
        if config['pytorch']['resize']['train']:
            transforms_train_list.append(transforms.Resize(size=(config['pytorch']['resize']['train_size'], config['pytorch']['resize']['train_size'])))
        if config['pytorch']['centerCrop']['train']:
            transforms_train_list.append(transforms.Resize(size=(config['pytorch']['centerCrop']['train_size'], config['pytorch']['centerCrop']['train_size'])))
        if config['pytorch']['colorJitter']['train']:
            transforms_train_list.append(transforms.RandomApply([transforms.ColorJitter(config['pytorch']['colorJitter']['brightness'])], p=0.75))
        if config['pytorch']['randomCrop']['train']:
            transforms_train_list.append(transforms.RandomCrop(size=config['pytorch']['randomCrop']['train_size'], padding=config['pytorch']['randomCrop']['padding']))
        if config['pytorch']['randomResizedCrop']['train']:
            transforms_train_list.append(transforms.RandomResizedCrop(size=config['pytorch']['randomResizedCrop']['train_size']))
        if config['pytorch']['randomHorizontalFlip']['train']:
            transforms_train_list.append(transforms.RandomHorizontalFlip())
        if config['pytorch']['randomAffine']['train']:
            transforms_train_list.append(transforms.RandomAffine(degrees=config['pytorch']['randomAffine']['degrees'], scale=config['pytorch']['randomAffine']['scale']))
        if config['pytorch']['randomRotation']['train']:
            transforms_train_list.append(transforms.RandomRotation(degrees=config['pytorch']['randomRotation']['degrees']))
        if config['pytorch']['toTensor']['train']:
            transforms_train_list.append(transforms.ToTensor())
        if config['pytorch']['normalize']['train']:
            transforms_train_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        if config['pytorch']['randomErasing']['train']:
            transforms_train_list.append(transforms.RandomErasing(p=config['pytorch']['randomErasing']['p'], value=config['pytorch']['randomErasing']['value']))
        train_transforms = transforms.Compose(transforms_train_list)
    else:
        if config['albumentations']['resize']['train']:
            transforms_train_list.append(A.Resize(config['albumentations']['resize']['train_size'], config['albumentations']['resize']['train_size']))
        if config['albumentations']['centerCrop']['train']:
            transforms_train_list.append(transforms.Resize(config['albumentations']['centerCrop']['train_size'], config['albumentations']['centerCrop']['train_size']))
        if config['albumentations']['horizontalFlip']['train']:
            transforms_train_list.append(A.HorizontalFlip())
        if config['albumentations']['rotate']['train']:
            transforms_train_list.append(A.Rotate(config['albumentations']['rotate']['limit']))
        if config['albumentations']['clahe']['train']:
            transforms_train_list.append(A.CLAHE())
        if config['albumentations']['gaussNoise']['train']:
            transforms_train_list.append(A.GaussNoise())
        if config['albumentations']['randomBrightness']['train']:
            transforms_train_list.append(A.RandomBrightness())
        if config['albumentations']['randomContrast']['train']:
            transforms_train_list.append(A.RandomContrast())
        if config['albumentations']['randomBrightnrssContrast']['train']:
            transforms_train_list.append(A.RandomBrightnessContrast())
        if config['albumentations']['hueSaturationValue']['train']:
            transforms_train_list.append(A.HueSaturationValue())
        if config['albumentations']['normalize']['train']:
            transforms_train_list.append(A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        if config['albumentations']['toTensor']['train']:
            transforms_train_list.append(AT.ToTensor())
        train_transforms = A.Compose(transforms_train_list)
        
    return train_transforms
                                
def get_transforms_test():
    transforms_test_list = []

    if config['transforms'] == 'pytorch':
        if config['pytorch']['resize']['test']:
            transforms_test_list.append(transforms.Resize(size=(config['pytorch']['resize']['test_size'], config['pytorch']['resize']['test_size'])))
        if config['pytorch']['centerCrop']['test']:
            transforms_test_list.append(transforms.Resize(size=(config['pytorch']['centerCrop']['test_size'], config['pytorch']['centerCrop']['test_size'])))
        if config['pytorch']['colorJitter']['test']:
            transforms_test_list.append(transforms.RandomApply([transforms.ColorJitter(config['pytorch']['colorJitter']['brightness'])], p=0.75))
        if config['pytorch']['randomCrop']['test']:
            transforms_test_list.append(transforms.RandomCrop(size=config['pytorch']['randomCrop']['test_size'], padding=config['pytorch']['randomCrop']['padding']))
        if config['pytorch']['randomResizedCrop']['test']:
            transforms_test_list.append(transforms.RandomResizedCrop(size=config['pytorch']['randomResizedCrop']['test_size']))
        if config['pytorch']['randomHorizontalFlip']['test']:
            transforms_test_list.append(transforms.RandomHorizontalFlip())
        if config['pytorch']['randomAffine']['test']:
            transforms_test_list.append(transforms.RandomAffine(degrees=config['pytorch']['randomAffine']['degrees'], scale=config['pytorch']['randomAffine']['scale']))
        if config['pytorch']['randomRotation']['test']:
            transforms_test_list.append(transforms.RandomRotation(degrees=config['pytorch']['randomRotation']['degrees']))
        if config['pytorch']['toTensor']['test']:
            transforms_test_list.append(transforms.ToTensor())
        if config['pytorch']['normalize']['test']:
            transforms_test_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        if config['pytorch']['randomErasing']['test']:
            transforms_test_list.append(transforms.RandomErasing(p=config['pytorch']['randomErasing']['p'], value=config['pytorch']['randomErasing']['value']))
        test_transforms = transforms.Compose(transforms_test_list)
    else:
        if config['albumentations']['resize']['test']:
            transforms_test_list.append(A.Resize(config['albumentations']['resize']['test_size'], config['albumentations']['resize']['test_size']))
        if config['albumentations']['centerCrop']['test']:
            transforms_test_list.append(transforms.Resize(config['albumentations']['centerCrop']['test_size'], config['albumentations']['centerCrop']['test_size']))
        if config['albumentations']['horizontalFlip']['test']:
            transforms_test_list.append(A.HorizontalFlip())
        if config['albumentations']['rotate']['test']:
            transforms_test_list.append(A.Rotate(config['albumentations']['rotate']['limit']))
        if config['albumentations']['clahe']['test']:
            transforms_test_list.append(A.CLAHE())
        if config['albumentations']['gaussNoise']['test']:
            transforms_test_list.append(A.GaussNoise())
        if config['albumentations']['randomBrightness']['test']:
            transforms_test_list.append(A.RandomBrightness())
        if config['albumentations']['randomContrast']['test']:
            transforms_test_list.append(A.RandomContrast())
        if config['albumentations']['randomBrightnrssContrast']['test']:
            transforms_test_list.append(A.RandomBrightnessContrast())
        if config['albumentations']['hueSaturationValue']['test']:
            transforms_test_list.append(A.HueSaturationValue())
        if config['albumentations']['normalize']['test']:
            transforms_test_list.append(A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        if config['albumentations']['toTensor']['test']:
            transforms_test_list.append(AT.ToTensor())
        test_transforms = A.Compose(transforms_test_list)
        
    return test_transforms                       
        
def get_train_data():
    train = pd.read_csv(os.path.join(config['previous_data_path'], 'trainLabels_cropped.csv'))
    y_train = train['level'].values
    
    valid = pd.read_csv(os.path.join(config['data_path'], 'train.csv'))
    y_valid = get_label_data(valid)
    
    train_dataset = TrainDataset(image=train['image'].values, transform=get_transforms_train(), y=y_train)
    valid_dataset = ValidDataset(id_code=valid['id_code'].values, transform=get_transforms_test(), y=y_valid)
#     tr, val = train_test_split(train['diagnosis'], stratify=train['diagnosis'], test_size=config['valid_size'])

#     train_sampler = SubsetRandomSampler(list(tr.index))
#     valid_sampler = SubsetRandomSampler(list(val.index))
                                
#     train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler, num_workers=config['num_workers'])
#     valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], sampler=valid_sampler, num_workers=config['num_workers'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])

    return train_loader, valid_loader
    
def get_test_data():
    test = pd.read_csv(os.path.join(config['data_path'], 'test.csv'))

    test_dataset = TestDataset(id_code=test['id_code'].values, transform=get_transforms_test())
    print(len(test_dataset)) # debug
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])
    
    return test_loader


def train(model, criterion, optimizer, train_loader):
    model.train()
    
    running_loss = 0
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(config['device']), target.to(config['device'])
        
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            output = model(data)
            data = data.view(-1, 1)
            target = target.view(-1, 1)
            loss = criterion(output.float(), target.float())
        
        running_loss += loss.data
        
        loss.backward()
        optimizer.step()
        
    return running_loss / len(train_loader)

def valid(model, criterion, optimizer, valid_loader):
    model.eval()
    
    running_loss = 0
    for _, (data, target) in enumerate(valid_loader):
        data, target = data.to(config['device']), target.to(config['device'])
        
        with torch.set_grad_enabled(False):
            output = model(data)
            data = data.view(-1, 1)
            target = target.view(-1, 1)
            loss = criterion(output.float(), target.float())
        
        running_loss += loss.data
        
    return running_loss / len(valid_loader)

def test(model, test_loader):
    model.eval()
    
    preds = []
    for (data, _, name) in test_loader:
        data = data.to(config['device'])
            
        output = model(data)
        output = output.cpu().detach().numpy()
        
        for o in output:
            preds.append(o)
    
    return preds


class DRModel(nn.Module):
    
    def __init__(self):
        
        super(DRModel, self).__init__()
        self.model = get_model()
        # self.model.load_state_dict(torch.load(config['weight_path']))
        if 'efficient' not in config['model_name']:
            self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=config['num_classes'], bias=True)
        else:
            self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=config['num_classes'], bias=True)
            
    def forward(self, x):
        
        x = self.model(x)
        if 'efficient' not in config['model_name']:
            x = F.softmax(x)
        
        return x


class TrainDataset(Dataset):
    
    def __init__(self, image, transform, y):
        
        self.image_name_list = [os.path.join(config['previous_data_path'], 'resized_train_cropped', 'resized_train_cropped', f'{image_name}.jpeg') for image_name in tqdm(image)]
#         self.image_list = [preprocess(image_name) for image_name in tqdm(self.image_name_list)]
        
        self.transform = transform
        self.labels = y
        
    def __len__(self):
        
        return len(self.image_name_list)
    
    def __getitem__(self, idx):
        
        image_name = self.image_name_list[idx]
        image = preprocess(image_name)
        
        if config['transforms'] == 'pytorch':
            image = transforms.ToPILImage()(image)
            image = self.transform(image)
        elif config['transforms'] == 'albumentations':
            image = self.transform(image=image)
            image = image['image']
            
        label = self.labels[idx]
        
        return image, label
    
class ValidDataset(Dataset):
    
    def __init__(self, id_code, transform, y):
        
        self.image_name_list = [os.path.join(config['data_path'], 'train_images', f'{image_name}.png') for image_name in tqdm(id_code)]
        
        self.transform = transform
        self.labels = np.zeros((len(self.image_name_list), config['num_classes']))
        
    def __len__(self):
        
        return len(self.image_name_list)

    def __getitem__(self, idx):
        
        image_name = self.image_name_list[idx]
        image = preprocess(image_name)
        
        if config['transforms'] == 'pytorch':
            image = transforms.ToPILImage()(image)
            image = self.transform(image)
        elif config['transforms'] == 'albumentations':
            image = self.transform(image=image)
            image = image['image']
            
        label = self.labels[idx]
        
        return image, label
    
class TestDataset(Dataset):
    
    def __init__(self, id_code, transform):
        
        self.image_name_list = [os.path.join(config['data_path'], 'test_images', f'{image_name}.png') for image_name in tqdm(id_code)]
        
        self.transform = transform
        self.labels = np.zeros((len(self.image_name_list), config['num_classes']))
        
    def __len__(self):
        
        return len(self.image_name_list)

    def __getitem__(self, idx):
        
        image_name = self.image_name_list[idx]
        image = preprocess(image_name)
        
        if config['transforms'] == 'pytorch':
            image = transforms.ToPILImage()(image)
            image = self.transform(image)
        elif config['transforms'] == 'albumentations':
            image = self.transform(image=image)
            image = image['image']
            
        label = self.labels[idx]
        
        return image, label, image_name


class EarlyStopping:

    def __init__(self):
 
        self.patience = config['patience']
        self.verbose = config['verbose']
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.epochs = 0

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.epochs += 1
        torch.save(model.state_dict(), f'best_{self.epochs}.pth')
        self.val_loss_min = val_loss


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        print(-loss_partial(self.coef_['x']))

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']


def main():
    seed_everything()

    model = DRModel().to(config['device'])
    train_loader, valid_loader = get_train_data()

    criterion = get_loss()
    optimizer = get_optimizer(params=model.parameters())
    early_stopping = EarlyStopping()

    for epoch in tqdm(range(config['epochs'])):
        train_loss = train(model, criterion, optimizer, train_loader)
        val_loss = valid(model, criterion, optimizer, valid_loader)

        print('epoch {:d}, loss: {:.4f} val_loss: {:.4f}'.format(epoch, train_loss, val_loss))

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    del train_loader
    gc.collect()

    for epoch in tqdm(range(5)):
        train_loss = train(model, criterion, optimizer, valid_loader)

    del valid_loader
    gc.collect()

    test_loader = get_test_data()

    coefficients=[0.5, 1.5, 2.5, 3.5]
    opt = OptimizedRounder()
    preds = test(model, test_loader)
    test_pred = opt.predict(preds, coefficients)
    test_df = pd.read_csv(os.path.join(config['data_path'], 'sample_submission.csv'))
    test_df['diagnosis'] = test_pred.astype(int)
    test_df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()
