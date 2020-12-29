"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import cv2
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import os


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

        if 'clip' in opt.preprocess:    # Set dataset dependent variables for preprocessing.
            self.clip_values = self.get_clip_values() 
        else:
            self.clip_values = None
        
        if 'minmax' in opt.preprocess:
            if 'clip' in opt.preprocess:
                self.minmax_values = self.clip_values
            else:
                self.minmax_values = self.get_minmax_values() 
        else:
            self.minmax_values = None
        

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass
    
    def get_clip_values(self):

        first_folder = os.path.join(self.root, os.listdir(self.root)[0])
        first_file = os.listdir(first_folder)[0]

        temp = Image.open(os.path.join(first_folder,first_file))
        nump = np.array(temp)
        hist = np.unique(nump)

        for folder in os.listdir(self.root):

            files = os.listdir(os.path.join(self.root, folder))

            for f in files:
                temp = Image.open(os.path.join(self.root,folder, f))
                nump = np.ndarray.flatten(np.array(temp))
                combined = np.concatenate((hist, nump))
                hist = np.unique(combined)

        bottom_value = hist[int(len(hist)*0.1)]
        top_value = hist[int(len(hist)*(1-0.1))]

        return top_value, bottom_value 

    def get_minmax_values(self):

        first_folder = os.path.join(self.root, os.listdir(self.root)[0])
        first_file = os.listdir(first_folder)[0]

        temp = Image.open(os.path.join(first_folder,first_file))
        nump = np.array(temp)
        min = np.min(nump)
        max = np.max(nump)

        for folder in os.listdir(self.root):

            files = os.listdir(os.path.join(self.root, folder))

            for f in files:
                temp = Image.open(os.path.join(self.root,folder, f))
                nump = np.ndarray.flatten(np.array(temp))
                min = min if min < np.min(nump) else np.min(nump)
                max = max if max > np.max(nump) else np.max(nump)
                
        return max, min 



def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True, clip_values = None, minmax_values = None, A = True):
    transform_list = []
    #import pdb; pdb.set_trace()
    
    if 'keep_values' in opt.preprocess:
        grayscale = convert = False

    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    
    if 'clip' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __clip(img, clip_values)))
    if 'minmax' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __minmax(img, minmax_values)))
    
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    
    if not A and 'sobel' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __sobel(img)))


    if 'keep_values' in opt.preprocess:
        transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)

def __sobel(image):
    
    Nimg = np.array(image)

    sobXImg = cv2.Sobel(Nimg, cv2.CV_64F, 1, 0, ksize=3)

    sobYImg = cv2.Sobel(Nimg, cv2.CV_64F, 0, 1, ksize=3)

    sobMagnitude = np.sqrt(np.power(sobXImg,2) + np.power(sobYImg,2))

    return Image.fromarray(sobMagnitude)    # .convert(mode='L')


def __clip(img, clip_values):
    return Image.fromarray(np.clip(img, clip_values[1], clip_values[0]))

def __minmax(img, minmax_values):
    return Image.fromarray((np.array(img)-minmax_values[1])/(minmax_values[0] - minmax_values[1]))

def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
