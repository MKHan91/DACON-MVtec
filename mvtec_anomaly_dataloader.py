import cv2
import random
import numpy as np
import pandas as pd
import os.path as osp
import torchvision.transforms as transforms

from glob import glob
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image


class mvtecDatasetPreprocess:
    def __init__(self, data_dir, mode='train'):
        if mode == 'train':
            self.image_paths = glob(osp.join(data_dir, mode, "*.png"))
            csv_path         = osp.join(data_dir, 'train_df.csv')
            self.labels      = self.read_label_csv(csv_path)
        
        elif mode == 'test':
            self.image_paths  = glob(osp.join(data_dir, mode, '*.png'))
        
        
    def read_label_csv(self, csv_path):
        train_csv = pd.read_csv(csv_path)
        
        train_labels = train_csv['label']
        label_unique = sorted(np.unique(train_labels))
        label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}

        train_labels = [label_unique[k] for k in train_labels]
        
        return train_labels



class mvtecDataset(Dataset):
    def __init__(self, image_paths, labels, mode='train'):
        self.image_paths = image_paths
        self.labels      = labels
        self.mode        = mode
        
        
    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path).convert('RGB')

        # 데이터 증강
        if self.mode=='train':
            data_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),             
                transforms.RandomVerticalFlip(),               
                transforms.RandomRotation(45),                 
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
                transforms.RandomResizedCrop(512),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
            image = data_transforms(image)
        
        if self.mode=='test':
            pass

        label = self.labels[idx]
        
        return image, label


    def get_labels(self):
        return self.labels