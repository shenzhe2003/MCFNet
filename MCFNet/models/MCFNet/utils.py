from PIL import Image, ImageOps, ImageFilter
import platform, os
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import  torch
from torch.nn import init
from datetime import datetime
import argparse
import shutil
from  matplotlib import pyplot as plt
from .Modules.DEConv import *

class TrainSet(Dataset):
    NUM_CLASS = 1
    def __init__(self, dataset_dir, img_id , base_size=512, crop_size=256, transform=None, suffix='.png'):
        super(TrainSet, self).__init__()
        self.dataset_dir = dataset_dir 
        self.transform = transform
        self._items = img_id                             
        self.masks = dataset_dir+'/'+'masks'              
        self.images = dataset_dir+'/'+'images'          
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix

    def _sync_transform(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT) 
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR) 
        mask = mask.resize((ow, oh), Image.NEAREST) 
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        img_1 = img 
        img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        img,img_1, mask = np.array(img), np.array(img_1), np.array(mask, dtype=np.float32)
        return img, img_1, mask

    def __getitem__(self, idx):

        img_id     = self._items[idx]                        
        img_path   = self.images+'/'+img_id+self.suffix  
        label_path = self.masks +'/'+img_id+self.suffix

        img = Image.open(img_path).convert('RGB')            
        mask = Image.open(label_path)

        img, img_1, mask = self._sync_transform(img, mask)

        if self.transform is not None:
            img = self.transform(img)
            img_1 = self.transform(img_1)
        mask = np.expand_dims(mask, axis=0).astype('float32')/ 255.0


        return img, img_1, torch.from_numpy(mask)


    def __len__(self):
        return len(self._items)

class TestSet(Dataset):
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id,transform=None,base_size=512,crop_size=256,suffix='.png'):
        super(TestSet, self).__init__()

        self.dataset_dir = dataset_dir
        self.transform = transform
        self._items    = img_id
        self.masks     = dataset_dir+'/'+'masks'
        self.images    = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        
        long_size = random.randint(int(self.base_size * 0.8), int(self.base_size * 1.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR) 
        mask = mask.resize((ow, oh), Image.NEAREST) 
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        img, mask = np.array(img), np.array(mask, dtype=np.float32) 
        return img, mask

    def _test_sync_transform(self, img, mask):
        img  = img.resize ((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)
        crop_size = self.crop_size
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop 
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        img, mask = np.array(img), np.array(mask, dtype=np.float32)  # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img, mask
    
    def __getitem__(self, idx):
        img_id = self._items[idx] 
        img_path   = self.images+'/'+img_id+self.suffix   
        label_path = self.masks +'/'+img_id+self.suffix

        img  = Image.open(img_path).convert('RGB')  
        mask = Image.open(label_path)
        img, mask = self._test_sync_transform(img, mask)

        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0

        return img, torch.from_numpy(mask)

    def __len__(self):
        return len(self._items)

class DemoSet (Dataset):
    NUM_CLASS = 1

    def __init__(self, dataset_dir, transform=None,base_size=512,crop_size=256,suffix='.png'):
        super(DemoSet, self).__init__()
        self.transform = transform
        self.images    = dataset_dir
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _demo_sync_transform(self, img):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)

        img = np.array(img)
        return img

    def img_preprocess(self):
        img_path   =  self.images
        img  = Image.open(img_path).convert('RGB')
        img  = self._demo_sync_transform(img)
        if self.transform is not None:
            img = self.transform(img)

        return img

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 and hasattr(m, 'weight'):
        init.xavier_normal_(m.weight.data)
    if isinstance(m, (Conv2d_cd,Conv2d_ad,Conv2d_hd,Conv2d_vd)):
        init.xavier_normal_(m.conv.weight.data)
        if m.conv.bias is not None:
            init.zeros_(m.conv.bias.data)
   
           

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def init_weights(net, init_type='normal'):
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

