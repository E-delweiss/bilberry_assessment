import os, sys
import glob
import pandas as pd
import random as rd

import numpy as np
import PIL

import torch
import torchvision


class BilberryDataset(torch.utils.data.Dataset):
    def __init__(self, ratio:int, isValSet_bool:bool=None, isAugment_bool:bool=False, isNormalize_bool:bool=False)->tuple: 
        super(BilberryDataset, self).__init__()

        self.isNormalize_bool = isNormalize_bool
        self.isAugment_bool = isAugment_bool
        self.ratio = ratio

        ### Load image paths
        imgset_roads = glob.glob('dataset/roads/*')
        imgset_fields = glob.glob('dataset/fields/*')
        len_roads = len(imgset_roads)
        len_fields = len(imgset_fields)
        assert len_roads > 0, "Error in loading road images."
        assert len_fields > 0, "Error in loading field images."

        ### Set validation dataset as 30% of the whole dataset
        if isValSet_bool:
            self.imgset_roads = rd.sample(imgset_roads, int(len_roads*0.3)) 
            self.imgset_fields = rd.sample(imgset_fields, int(len_fields*0.3))
        else :
            self.imgset_roads = rd.sample(imgset_roads, int(len_roads*0.7)) 
            self.imgset_fields = rd.sample(imgset_fields, int(len_fields*0.7))

        self.label_roads = np.zeros(len(self.imgset_roads)).tolist()
        self.label_fields = np.ones(len(self.imgset_fields)).tolist()
        
        self.labelset = self.label_roads + self.label_fields
        self.imgset = self.imgset_roads + self.imgset_fields


    def _preprocess(self, img_PIL:torch.Tensor)->torch.Tensor:
        ### Resizing
        augment = torchvision.transforms.Resize((300,300)) ### ????
        img_t = augment(img_PIL)

        ### Data augmentation
        if self.isAugment_bool:
            augment = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop((224, 224)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ColorJitter(brightness=[0.5,2], contrast=[0.5,2.5], saturation=[0.5,1]),
                torchvision.transforms.RandomAdjustSharpness(5, p=0.5),
                torchvision.transforms.ToTensor()
                ])
            img_t = augment(img_t)
        
        ### Normalize data
        if self.isNormalize_bool:
            mean, std = (0.4551, 0.4672, 0.4151), (0.2522, 0.2451, 0.2808)
            img_t = torchvision.transforms.Normalize(mean, std)(img_t)

        return img_t

    def __len__(self):
        return len(self.imgset)
    
    def __getitem__(self, idx)->tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ### Trick to assure 1:1 field vs road imgs ratio 
        if self.ratio:
            pos_idx = idx // (self.ratio + 1)

            ### TODO
            if idx % (self.ratio + 1):
                neg_idx = idx - 1 - pos_idx
                neg_idx = neg_idx % len(self.imgset_roads)
                img_path = self.imgset_roads[neg_idx]
                label = self.label_roads[neg_idx]
            else:
                pos_idx = pos_idx % len(self.imgset_fields)
                img_path = self.imgset_fields[pos_idx]
                label = self.label_fields[pos_idx]
        else:
            img_path = self.imgset[idx]
            label = self.labelset[idx]

        img_path = self.imgset[idx]
        img_PIL = PIL.Image.open(img_path).convert('RGB')
        image = self._preprocess(img_PIL)

        label = self.labelset[idx]
        return image, torch.tensor(label).to(torch.int32)


def get_training_dataset(BATCH_SIZE=16, **kwargs):
    """
    Loads and maps the training split of the dataset using the custom dataset class. 
    """
    dataset = BilberryDataset(isValSet_bool=False, **kwargs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

def get_validation_dataset(BATCH_SIZE=None, **kwargs):
    """
    Loads and maps the validation split of the datasetusing the custom dataset class. 
    """
    dataset = BilberryDataset(isValSet_bool=True, **kwargs)
    if BATCH_SIZE is None:
        BATCH_SIZE = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


if __name__ == "__main__":
    dataset = get_training_dataset(ratio=1, isAugment_bool=True, isNormalize_bool=False)
    print(next(iter(dataset))[0])