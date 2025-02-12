# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import cv2
import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import random

try:
    import pyspng
except ImportError:
    pyspng = None

from pycocotools.coco import COCO
from pathlib import Path
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageNet
from datasets.mask_generator_256 import RandomMask

def multiload(data, L):
    ans = data.annToMask(data.loadAnns(L[0])[0])
    for i in range(1,len(L)):
        ans += data.annToMask(data.loadAnns(L[i])[0])
    return ans

class PartImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, use_labels=None, crop_size=256, max_size=256, xflip=False, random_seed=0):
        val_annfile = '/raid/student/2021/ai21btech11005/PartImageNet/annotations/test/test.json'
        self.maskroot = Path(val_annfile).parent.parent/Path(val_annfile).stem
        self.imageroot = self.maskroot.parent.parent/'images'/ Path(val_annfile).stem
        self.data = COCO(val_annfile)   
        self.wnid_to_idx = ImageNet('/raid/student/2021/ai21btech11005/Imagenet', split='val').wnid_to_idx
        self.image_shape = (3, crop_size, crop_size)
        self.name = 'PartImageNet'
        # transform_list = [
        #         transforms.Resize((crop_size, crop_size), 
        #             interpolation=Image.BICUBIC),
        #         transforms.ToTensor(), 
        #         transforms.Normalize((0.485, 0.456, 0.406),
        #                                         (0.229, 0.224, 0.225))
        #         # transforms.Normalize((0.5, 0.5, 0.5),
        #         #                     (0.5, 0.5, 0.5))
        #         ]
        # self.image_transform = transforms.Compose(transform_list)
        # self.mask_transform = transforms.Compose([
        #     transforms.Resize((crop_size, crop_size),interpolation=Image.BICUBIC),
        #     transforms.ToTensor()
        #     ])     
                
        # self.deny_indices = [21, 85, 145, 380, 969, 977, 1158]  # For val
        self.deny_indices = [65, 347, 533, 657, 728, 945, 969, 1061, 1613, 1630, 1738, 1819, 1906, 1928, 2001, 2219, 2254, 2406]  #For test
        self.allow_indices = [i for i in range(len(self.data.dataset['images'])) if i not in self.deny_indices]

    def __len__(self):
        return len(self.allow_indices)
    
    def __getitem__(self, idx):
        try:
            idx = self.allow_indices[idx]
        except:
            print("Error accesing", idx, "Limit: ", len(self.allow_indices))
        file_name = Path(self.data.loadImgs(idx)[0]['file_name'])
        image = Image.open(str(self.imageroot/file_name)).resize((256, 256), Image.BICUBIC).convert('RGB')
        
        mask = multiload(self.data, self.data.getAnnIds(imgIds=idx))
        
        mask = np.where(mask>0,1,0)
        mask = Image.fromarray((mask*255).astype(np.uint8)).resize((256, 256), Image.BICUBIC)
                
        label = self.wnid_to_idx[str(file_name).split('_')[0]]
        one_hot = np.zeros(1000)
        one_hot[label] = 1        

        #Convert into NCHW format
        image = np.array(image).transpose(2, 0, 1)        
        mask = np.expand_dims(np.array(mask),0)
        return image, mask, one_hot
    
    def get_label(self, idx):
        return self.__getitem__(idx)[2]
    
    @property
    def resolution(self):
        return 256
    
    @property
    def has_labels(self):
        return True


class FilteredImageNet(torch.utils.data.Dataset):
    def __init__(self, imagenet_dir, classes_file, partition, val_annfile):
        self.classes = []        
        with open(classes_file, 'r') as f:
            for line in f:
                self.classes.append(line.strip())        
        self.train_dataset = ImageNet(root=imagenet_dir, split=partition)
        self.classes = [self.train_dataset.wnid_to_idx[i] for i in self.classes]
        self.filtered_indices = [i for i, target in enumerate(self.train_dataset.targets) if target in self.classes]
        self.filtered_indices_map = {Path(self.train_dataset.imgs[i][0]).stem:i for i in self.filtered_indices}

        partinet = PartImageNetDataset(val_annfile)
        for i in range(len(partinet)):
            self.filtered_indices_map.pop(Path(partinet.data.dataset['images'][i]['file_name']).stem, None)
                        
        self.filtered_indices = [self.filtered_indices_map[k] for k in self.filtered_indices_map]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        image, label = self.train_dataset[self.filtered_indices[idx]]
        # Resize image to 256x256
        image = image.resize((256, 256), Image.BICUBIC)
        return image, label
    
    

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        # return any(x != 0 for x in self.label_shape)
        return True

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


#----------------------------------------------------------------------------


class ImageFolderMaskDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        hole_range=[0,1],
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self._hole_range = hole_range

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC

        # for grayscale image
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        # restricted to 256x256
        res = 256
        H, W, C = image.shape
        if H < res or W < res:
            top = 0
            bottom = max(0, res - H)
            left = 0
            right = max(0, res - W)
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)
        H, W, C = image.shape
        h = random.randint(0, H - res)
        w = random.randint(0, W - res)
        image = image[h:h+res, w:w+res, :]

        image = np.ascontiguousarray(image.transpose(2, 0, 1)) # HWC => CHW

        return image

    def _load_raw_labels(self):
        fname = 'labels.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])

        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        mask = RandomMask(image.shape[-1], hole_range=self._hole_range)  # hole as 0, reserved as 1
        return image.copy(), mask, self.get_label(idx)


class ModImageFolderMaskDataset(Dataset):
    def __init__(self,
        # path,     
        use_labels=None,              # Path to directory or zip.
        resolution      = 256, # Ensure specific resolution, None = highest available.
        hole_range=[0,1],
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):        
        self._zipfile = None
        self._hole_range = hole_range
        self.res = resolution
        # self.path = '/raid/student/2021/ai21btech11005/Imagenet'

        # if os.path.isdir(self._path):
        #     self._type = 'dir'
        #     self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        # elif self._file_ext(self._path) == '.zip':
        #     self._type = 'zip'
        #     self._all_fnames = set(self._get_zipfile().namelist())
        # else:
        #     raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        # self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        # if len(self._image_fnames) == 0:
        #     raise IOError('No image files found in the specified path')
        self.dataset = FilteredImageNet('/raid/student/2021/ai21btech11005/Imagenet',
                           '/raid/student/2021/ai21btech11005/co-mod-gan-pytorch/PIN_classes.txt',
                           'train',
                           '/raid/student/2021/ai21btech11005/PartImageNet/annotations/test/test.json')

        # name = os.path.splitext(os.path.basename(self._path))[0]
        name = "ImageNet"
        raw_shape = [len(self.dataset)] + list(self._load_raw_image(0).shape)
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')
        
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        # fname = self._image_fnames[raw_idx]
        # with self._open_file(fname) as f:
        #     if pyspng is not None and self._file_ext(fname) == '.png':
        #         image = pyspng.load(f.read())
        #     else:
        #         image = np.array(PIL.Image.open(f))
        image = np.array(self.dataset[raw_idx][0])
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC

        # for grayscale image
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        # restricted to 256x256
        res = 256
        H, W, C = image.shape
        if H < res or W < res:
            top = 0
            bottom = max(0, res - H)
            left = 0
            right = max(0, res - W)
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)
        H, W, C = image.shape
        h = random.randint(0, H - res)
        w = random.randint(0, W - res)
        image = image[h:h+res, w:w+res, :]

        image = np.ascontiguousarray(image.transpose(2, 0, 1)) # HWC => CHW

        return image


    def _load_raw_labels(self):
        fname = 'labels.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        label = self.dataset[idx][1]
        one_hot = np.zeros(1000)
        one_hot[label] = 1
        one_hot_label = one_hot

        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        mask = RandomMask(image.shape[-1], hole_range=self._hole_range)  # hole as 0, reserved as 1
        return image.copy(), mask, one_hot_label



if __name__ == '__main__':
    res = 256
    dpath = '/data/liwenbo/datasets/Places365/standard/val_256'
    D = ImageFolderMaskDataset(path=dpath)
    print(D.__len__())
    for i in range(D.__len__()):
        print(i)
        a, b, c = D.__getitem__(i)
        if a.shape != (3, 256, 256):
            print(i, a.shape)
