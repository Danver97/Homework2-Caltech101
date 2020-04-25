from torchvision.datasets import VisionDataset

from PIL import Image

import pandas as pd

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        self.caltechDS = pd.read_csv('train.txt', header=None)
        self.caltechDS['img_paths'] = self.caltechDS[0]
        self.caltechDS = self.caltechDS.drop(0, axis=1)
        
        self.caltechDS['images'] = self.caltechDS.apply(lambda r: pil_loader(r['img_paths']), axis=1)
        self.caltechDS['class'] = self.caltechDS.apply(lambda r: r['img_paths'].split('/')[0], axis=1)

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image = self.caltechDS.loc[index, 'images']
        label = self.caltechDS.loc[index, 'class']

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.caltechDS.index)
        return length
