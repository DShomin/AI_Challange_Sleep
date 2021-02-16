from conf import *

import math
import random
from PIL import Image, ImageOps
# from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop, RandomApply, Resize, CenterCrop, RandomAffine
# from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomGrayscale, RandomRotation
import albumentations
from albumentations.pytorch import ToTensorV2
from albumentations import DualTransform
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur

import cv2

def get_transform(
        target_size=256,
        transform_list='horizontal_flip', # random_crop | keep_aspect
        # augment_ratio=0.5,
        is_train=True,
        ):
    transform = list()
    transform_list = transform_list.split(', ')
    # augments = list()

    
    for transform_name in transform_list:
        # default resize
        transform.append(Albumentations.Resize(height=target_size, width=target_size,p=1))

        if transform_name == 'random_crop':
            # scale = (0.6, 1.0) if is_train else (0.8, 1.0)
            transform.append(Albumentations.RandomResizedCrop(height=target_size, width=target_size,p=1))
        # elif transform_name == 'resize':
        #     transform.append(Resize(target_size))
        elif transform_name == 'horizontal_flip':
            transform.append(Albumentations.HorizontalFlip(p=0.5))
        elif transform_name == 'vertical_flip':
            transform.append(Albumentations.VerticalFlip(p=0.5))
        elif transform_name == 'griddropout':
            transform.append(Albumentations.GridDropout())


    # transform.append(RandomApply(augments, p=augment_ratio))   
    transform.append(ToTensorV2())
    transform.append(Albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return Albumentations.Compose(transform)


def create_train_transforms(args, size=224):
    if args.model=='vit_small_patch16_224':
        return Compose([albumentations.Resize(224, 224)])
    else:
        translist = []
        if args.gaussnoise:
            #translist += [albumentations.GaussNoise(var_limit=(5.0, 30.0), p=0.4)]
            translist += [albumentations.OneOf([
                                                albumentations.MotionBlur(blur_limit=5),
                                                albumentations.MedianBlur(blur_limit=5),
                                                #albumentations.GaussianBlur(blur_limit=5),
                                                albumentations.GaussNoise(var_limit=(5.0, 30.0))], p=args.noise_p)]
        if args.clahe:
            translist += [albumentations.CLAHE(clip_limit=4.0, p=0.1)]
        if args.cutout:
            translist += [albumentations.Cutout(max_h_size=int(270 * 0.375), max_w_size=int(480 * 0.375), num_holes=1, p=0.8)]

        transform = Compose(translist)
        return transform

def create_val_transforms(args, size=224):
    if args.model=='vit_small_patch16_224':
        return Compose([albumentations.Resize(224, 224)])
    else:
        return Compose([
            # IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            #albumentations.Resize(384, 384), 
            #albumentations.HorizontalFlip(),
            ])



class IsotropicResize(DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized
