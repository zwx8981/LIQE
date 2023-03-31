import os
import torch
import functools
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset

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

type2label = {'jp2k':0, 'jpeg':1, 'wn':2, 'gblur':3, 'fastfading':4}
content2label ={'animal':0, 'cityscape':1, 'human':2, 'indoor':3, 'landscape':4, 'night':5, 'plant':6, 'still_life':7, 'others':8}

class ImageDataset(Dataset):
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
        batch_size = 1
        n_channels = 3
        n_rows = I.size(2)
        n_cols = I.size(3)
        kernel_h = 224
        kernel_w = 224
        step = 32
        #x = torch.arange(batch_size * n_channels * n_rows * n_cols).view(batch_size, n_channels, n_rows, n_cols)
        # unfold(dimension, size, step)
        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1,
                                                                                                          n_channels,
                                                                                                          kernel_h,
                                                                                                          kernel_w)

        assert patches.size(0) >= self.num_patch
        #self.num_patch = np.minimum(patches.size(0), self.num_patch)
        sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
        patches = patches[sel, ...]
        mos = self.data.iloc[index, 1]

        dist_type = self.data.iloc[index, 2]
        scene_content1 = self.data.iloc[index, 3]
        scene_content2 = self.data.iloc[index, 4]
        scene_content3 = self.data.iloc[index, 5]

        scene_text1 = 'a photo of a ' + scene_content1
        scene_text2 = 'a photo of a ' + scene_content2
        scene_text3 = 'a photo of a ' + scene_content3
        dist_text = 'a photo with ' + dist_type + ' artifacts'

        if scene_content2 == 'invalid':
            scene_content = scene_content1
            scene_text = scene_text1
            valid = 1
        elif scene_content3 == 'invalid':
            sel = np.random.randint(2, size=1)
            if sel == 0:
                scene_content = scene_content1
                scene_text = scene_text1
            else:
                scene_content = scene_content2
                scene_text = scene_text2
            valid = 2
        else:
            sel = np.random.randint(3, size=1)
            if sel == 0:
                scene_content = scene_content1
                scene_text = scene_text1
            elif sel == 1:
                scene_content = scene_content2
                scene_text = scene_text2
            else:
                scene_content = scene_content3
                scene_text = scene_text3
            valid = 3

        if not self.test:
            sample = {'I': patches, 'mos': float(mos), 'dist_type': dist_type, 'dist_sentence': dist_text,
                      'scene_content': scene_content, 'scene_sentence': scene_text}
        else:
            sample = {'I': patches, 'mos': float(mos), 'dist_type': dist_type, 'dist_sentence': dist_text,
                      'scene_content1': scene_content1, 'scene_sentence1': scene_text1,
                      'scene_content2': scene_content2, 'scene_sentence2': scene_text2,
                      'scene_content3': scene_content3, 'scene_sentence3': scene_text3,
                      'valid': valid}

        return sample

    def __len__(self):
        return len(self.data.index)

class ImageDataset_Inf(Dataset):
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

        if not test:
            self.data = self.data.sample(frac=1)
            for i in range(99999):
                data_t = self.data.sample(frac=1)
                self.data = pd.concat([self.data, data_t], axis=0)

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
        batch_size = 1
        n_channels = 3
        n_rows = I.size(2)
        n_cols = I.size(3)
        kernel_h = 224
        kernel_w = 224
        step = 32
        #x = torch.arange(batch_size * n_channels * n_rows * n_cols).view(batch_size, n_channels, n_rows, n_cols)
        # unfold(dimension, size, step)
        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1,
                                                                                                          n_channels,
                                                                                                          kernel_h,
                                                                                                          kernel_w)

        assert patches.size(0) >= self.num_patch
        #self.num_patch = np.minimum(patches.size(0), self.num_patch)
        sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
        patches = patches[sel, ...]
        mos = self.data.iloc[index, 1]

        dist_type = self.data.iloc[index, 2]
        scene_content1 = self.data.iloc[index, 3]
        scene_content2 = self.data.iloc[index, 4]
        scene_content3 = self.data.iloc[index, 5]

        scene_text1 = 'a photo of a ' + scene_content1
        scene_text2 = 'a photo of a ' + scene_content2
        scene_text3 = 'a photo of a ' + scene_content3
        dist_text = 'a photo with ' + dist_type + ' artifacts'

        if scene_content2 == 'invalid':
            scene_content = scene_content1
            scene_text = scene_text1
            valid = 1
        elif scene_content3 == 'invalid':
            sel = np.random.randint(2, size=1)
            if sel == 0:
                scene_content = scene_content1
                scene_text = scene_text1
            else:
                scene_content = scene_content2
                scene_text = scene_text2
            valid = 2
        else:
            sel = np.random.randint(3, size=1)
            if sel == 0:
                scene_content = scene_content1
                scene_text = scene_text1
            elif sel == 1:
                scene_content = scene_content2
                scene_text = scene_text2
            else:
                scene_content = scene_content3
                scene_text = scene_text3
            valid = 3

        if not self.test:
            sample = {'I': patches, 'mos': float(mos), 'dist_type': dist_type, 'dist_sentence': dist_text,
                      'scene_content': scene_content, 'scene_sentence': scene_text}
        else:
            sample = {'I': patches, 'mos': float(mos), 'dist_type': dist_type, 'dist_sentence': dist_text,
                      'scene_content1': scene_content1, 'scene_sentence1': scene_text1,
                      'scene_content2': scene_content2, 'scene_sentence2': scene_text2,
                      'scene_content3': scene_content3, 'scene_sentence3': scene_text3,
                      'valid': valid}

        return sample

    def __len__(self):
        return len(self.data.index)



class ImageDataset_SPAQ(Dataset):
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
        self.data = pd.read_excel(csv_file)
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
        batch_size = 1
        n_channels = 3
        n_rows = I.size(2)
        n_cols = I.size(3)
        kernel_h = 224
        kernel_w = 224
        step = 32
        #x = torch.arange(batch_size * n_channels * n_rows * n_cols).view(batch_size, n_channels, n_rows, n_cols)
        # unfold(dimension, size, step)
        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1,
                                                                                                          n_channels,
                                                                                                          kernel_h,
                                                                                                          kernel_w)

        assert patches.size(0) >= self.num_patch
        #self.num_patch = np.minimum(patches.size(0), self.num_patch)
        sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
        patches = patches[sel, ...]
        mos = self.data.iloc[index, 1]

        sample = {'I': patches, 'mos': float(mos)}

        return sample

    def __len__(self):
        return len(self.data.index)


class ImageDataset_TID(Dataset):
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
        #self.data = pd.read_csv(csv_file, sep=' ', header=None)
        self.data = pd.read_csv(csv_file)
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
        #image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
        filename = self.data.iloc[index, 0]
        if filename[4:8] == '01_1':
            filename = 'I' + filename[1:]
        elif (filename[4:8] == '11_1') | (filename[4:8] == '13_1'):
            filename = 'I' + filename[1:-3] + 'BMP'
        image_name = os.path.join(self.img_dir, filename)
        I = self.loader(image_name)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        batch_size = 1
        n_channels = 3
        n_rows = I.size(2)
        n_cols = I.size(3)
        kernel_h = 224
        kernel_w = 224
        step = 32
        #x = torch.arange(batch_size * n_channels * n_rows * n_cols).view(batch_size, n_channels, n_rows, n_cols)
        # unfold(dimension, size, step)
        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1,
                                                                                                          n_channels,
                                                                                                          kernel_h,
                                                                                                          kernel_w)

        assert patches.size(0) >= self.num_patch
        #self.num_patch = np.minimum(patches.size(0), self.num_patch)
        sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
        patches = patches[sel, ...]
        mos = self.data.iloc[index, 1]

        sample = {'I': patches, 'mos': float(mos)}

        return sample

    def __len__(self):
        return len(self.data.index)


class ImageDataset_PIPAL(Dataset):
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
        #self.data = pd.read_csv(csv_file, sep=' ', header=None)
        self.data = pd.read_csv(csv_file)
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
        filename = self.data.iloc[index, 1]
        image_name = os.path.join(self.img_dir, filename)
        I = self.loader(image_name)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        batch_size = 1
        n_channels = 3
        n_rows = I.size(2)
        n_cols = I.size(3)
        kernel_h = 224
        kernel_w = 224
        step = 32
        #x = torch.arange(batch_size * n_channels * n_rows * n_cols).view(batch_size, n_channels, n_rows, n_cols)
        # unfold(dimension, size, step)
        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1,
                                                                                                          n_channels,
                                                                                                          kernel_h,
                                                                                                          kernel_w)

        assert patches.size(0) >= self.num_patch
        #self.num_patch = np.minimum(patches.size(0), self.num_patch)
        sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
        patches = patches[sel, ...]
        mos = self.data.iloc[index, 2]

        sample = {'I': patches, 'mos': float(mos)}

        return sample

    def __len__(self):
        return len(self.data.index)

class ImageDataset_ava(Dataset):
    def __init__(self, npy_file,
                 img_dir,
                 preprocess,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """

        self.data = np.load(npy_file, allow_pickle=True)
        print('%d csv data successfully loaded!' % len(self.data))
        self.img_dir = img_dir
        self.transform = preprocess
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = os.path.join(self.img_dir, self.data[index]['image'])
        I = self.loader(image_name)
        if self.transform is not None:
            I = self.transform(I)
        mos = self.data[index]['mean']
        sample = {'I': I, 'mos': mos}

        return sample
    def __len__(self):
        return len(self.data)