# -*- coding: utf-8 -*-

__author__ = 'kohou.wang'
__time__ = '19-9-24'
__email__ = 'oukohou@outlook.com'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.
# Well, I'm kidding... Always, Welcome to contact me.

"""Description for the script:
construct a dataloader for cleaned IMDB.
"""

import cv2
import os

from torch.utils.data import Dataset
from imgaug import augmenters as iaa
from torchvision import transforms as T


class IMDBDatasets(Dataset):
    def __init__(self, image_df, base_path, augment=False, mode='train'):
        self.images_df = image_df.copy()
        self.base_path = base_path
        self.mode = mode
        self.augment = augment
    
    def __len__(self):
        return len(self.images_df)
    
    def __getitem__(self, index):
        image_, image_path_ = self.read_images(index)
        if self.mode in ['train', ]:
            label = self.images_df.iloc[index].age
        else:
            label = image_path_
        if self.augment:
            image_ = self.augmentor(image_)
        
        image_ = T.Compose([
            T.ToPILImage(),
            # T.RandomResizedCrop(self.input_size),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(image_)
        return image_.float(), label
    
    def read_images(self, index_):
        filename = self.images_df.iloc[index_].Filename
        # import ipdb
        # ipdb.set_trace()
        image_path_ = os.path.join(self.base_path, filename)
        image = cv2.imread(image_path_)
        return image, image_path_
    
    def augmentor(self, image):
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.SomeOf((0, 4), [
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
            ]),
            iaa.OneOf([
                iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(2, 7)),  # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 11)),  # blur image using local medians with kernel sizes between 2 and 7
            ]),
            # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
        ], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug


if __name__ == "__main__":
    pass
