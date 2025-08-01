import random

import torch
import os.path
import PIL.Image as Image
import numpy as np
import torchvision.transforms.functional as F

from torchvision import transforms
from torch.utils import data

class DatasetCRACK(data.Dataset):

    def __init__(self, benchmark, datapath, split, img_mode, img_size):
        super(DatasetCRACK, self).__init__()
        self.split = 'val' if split in ['val', 'test'] else 'train.txt'
        self.benchmark = benchmark
        # assert self.benchmark == 'cracktree200' or self.benchmark == 'crack500' or self.benchmark == 'cracktav' or self.benchmark =='crackls315' or self.benchmark=='deepcrack'
        self.img_mode = img_mode
        assert img_mode in ['crop', 'same', 'resize']
        self.img_size = img_size

        self.img_path = os.path.join(datapath, 'images')
        self.ann_path = os.path.join(datapath, 'annotation_mask')
        self.bd_path = os.path.join(datapath, 'annotation_boundary')

        self.img_metadata = self.load_metadata()
        self.norm_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        if self.img_mode == 'resize':
            self.resize = transforms.Resize([img_size, img_size], interpolation=Image.NEAREST)
        else:
            self.resize = None

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, index):
        img_name = self.img_metadata[index]
        img, anno_mask, anno_boundary, org_img_size = self.load_frame(img_name)

        if self.split == 'train.txt':
            img, anno_mask, anno_boundary = self.augmentation(img, anno_mask, anno_boundary)

        if self.img_mode == 'resize' and self.split == 'train.txt':
            img = self.resize(img)
            anno_mask = self.resize(anno_mask)
            anno_boundary = self.resize(anno_boundary)
        elif self.img_mode == 'crop' and self.split == 'train.txt':
            i, j, h, w = self.get_params(img, (self.img_size, self.img_size))
            img = F.crop(img, i, j, h, w)
            anno_mask = F.crop(anno_mask, i, j, h, w)
            anno_boundary = F.crop(anno_boundary, i, j, h, w)
        else:
            pass
        #print(img.size)
        img = self.norm_img(img)

        batch = {
            'img_name': img_name,
            'img': img,
            'anno_mask': anno_mask,
            'anno_boundary': anno_boundary
        }
        return batch

    def augmentation(self, img, anno_mask, anno_boundary):
        p = np.random.choice([0, 1])
        transform_hflip = transforms.RandomHorizontalFlip(p)
        img = transform_hflip(img)
        anno_mask = transform_hflip(anno_mask)
        anno_boundary = transform_hflip(anno_boundary)

        p = np.random.choice([0, 1])
        transform_vflip = transforms.RandomVerticalFlip(p)
        img = transform_vflip(img)
        anno_mask = transform_vflip(anno_mask)
        anno_boundary = transform_vflip(anno_boundary)

        #img = self.to_tensor(img)
        if np.random.random() > 0.5:
            p = np.random.uniform(-180,180,1)[0]
            transform_rotate = transforms.RandomRotation((p, p), expand=True)
            img = transform_rotate(img)
            anno_mask = transform_rotate(anno_mask)
            anno_boundary = transform_rotate(anno_boundary)

        if np.random.random() > 0.5:
            color_aug = transforms.ColorJitter(brightness=[1.0, 2.1], contrast=[1.0, 2.1], saturation=[0.5, 1.5])
            #random_factor = np.random.randint(0, 31) / 10.
            #sharpe_aug = transforms.RandomAdjustSharpness(random_factor, p=0.5)
            img = color_aug(img)
            #img = sharpe_aug(img)

        return img, anno_mask, anno_boundary

    def load_frame(self, img_name):
        img = self.read_img(img_name)
        anno_mask = self.read_mask(img_name)
        anno_boundary = self.read_boundary(img_name)

        org_img_size = img.size

        return img, anno_mask, anno_boundary, org_img_size

    def read_mask(self, img_name):
        mask = np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png'))
        mask[mask == 0] = 0
        mask[mask == 255] = 1
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return mask

    def read_boundary(self, img_name):
        mask = np.array(Image.open(os.path.join(self.bd_path, img_name) + '.png'))
        mask[mask == 0] = 0
        mask[mask == 255] = 1
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return mask

    def read_img(self, img_name):
        # maybe png
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def load_metadata(self):
        if self.split == 'train.txt':
            meta_file = os.path.join('data/split', self.benchmark, 'train.txt')
            #meta_file = os.path.join(r'data\\split', self.benchmark, 'train.txt')
        elif self.split == 'val' or self.split == 'test':
            meta_file = os.path.join('data/split', self.benchmark, 'test.txt')
            #meta_file = os.path.join(r'data\\split', self.benchmark, 'test.txt')
        else:
            raise RuntimeError('Undefined split ', self.split)

        record_fd = open(meta_file, 'r')
        records = record_fd.readlines()

        img_metaname = [line.strip() for line in records]
        # print(img_metaname)
        # #exit(-1)

        return img_metaname

    def get_params(self, img, output_size):
        def _get_image_size(img):
            if F._is_pil_image(img):
                return img.size
            elif isinstance(img, torch.Tensor) and img.dim() > 2:
                return img.shape[-2:][::-1]
            else:
                raise TypeError('Unexpected type {}'.format(type(img)))

        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw