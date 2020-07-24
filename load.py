import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets,transforms, models
from collections import OrderedDict
from PIL import Image

#Loading Data

def get_data(image_dir,resize=256,center_crop=224,batch=64):
    data_dir = image_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = ['train_transforms','test_transforms']

    train_transforms = transforms.Compose([transforms.RandomRotation(20),
                                         transforms.RandomResizedCrop(center_crop),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(resize),
                                          transforms.CenterCrop(center_crop),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    image_datasets = ['train_data','valid_data','test_data']
    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir,transform=test_transforms)


    dataloaders = ['trainloader','validloader','testloader']
    trainloader = torch.utils.data.DataLoader(train_data,batch_size=batch,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data,batch_size=batch,shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data,batch_size=batch,shuffle=True)
    
    return train_data, trainloader,validloader,testloader