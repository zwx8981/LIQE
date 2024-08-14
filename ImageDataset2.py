import os
import torch
import functools
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

ImageFile.LOAD_TRUNCATED_IMAGES = True
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    return I.convert('RGB')


def get_default_img_loader():
    return functools.partial(image_loader)


class ImageDataset2(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 num_patch,
                 test,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file, sep='\t', header=None)
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
        I = self.loader(image_name)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 32
        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(0, 2, 3, 1, 4, 5).reshape(-1,
                                                                                                          n_channels,
                                                                                                          kernel_h,
                                                                                                          kernel_w)

        assert patches.size(0) >= self.num_patch
        #self.num_patch = np.minimum(patches.size(0), self.num_patch)
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
        patches = patches[sel, ...]
        mos = self.data.iloc[index, 1]

        dist_type = self.data.iloc[index, 2]
        scene_content1 = self.data.iloc[index, 3]
        scene_content2 = self.data.iloc[index, 4]
        scene_content3 = self.data.iloc[index, 5]

        if scene_content2 == 'invalid':
            valid = 1
        elif scene_content3 == 'invalid':
            valid = 2
        else:
            valid = 3

        sample = {'I': patches, 'mos': float(mos), 'dist_type': dist_type, 'scene_content1': scene_content1,
                  'scene_content2':scene_content2, 'scene_content3':scene_content3, 'valid':valid}

        return sample

    def __len__(self):
        return len(self.data.index)
        
class ImageDataset_qonly(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 num_patch,
                 set,
                 test,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        if csv_file[-3:] == 'txt':
            data = pd.read_csv(csv_file, sep='\t', header=None)
            self.data = data
        else:
            data = pd.read_csv(csv_file, header=0)
            self.data = data[data.split==set]
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
        I = self.loader(image_name)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 32
        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(0, 2, 3, 1, 4, 5).reshape(-1,
                                                                                                          n_channels,
                                                                                                          kernel_h,
                                                                                                          kernel_w)

        assert patches.size(0) >= self.num_patch
        #self.num_patch = np.minimum(patches.size(0), self.num_patch)
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
        patches = patches[sel, ...]
        mos = self.data.iloc[index, 1]


        sample = {'I': patches, 'mos': float(mos)}

        return sample

    def __len__(self):
        return len(self.data)

    def __len__(self):
        return len(self.data.index)
