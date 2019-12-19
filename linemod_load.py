import cv2
import numpy as np
import json
# import matplotlib.pyplot as plt
import glob
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import transform
import os
import copy

class LineModDataset(Dataset):
    """ Loading LineMod Dataset for Pose Estimation """

    def __init__(self, root_dir, task, object_number, transform=None):
        """
        Args:
            root_dir (string): Path to the LineMod dataset.
            object_number (int): Unique number of an object. (1, 2, ..., 15) -> converted into str (000001, 000002, ..., 000015)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        assert (task in ['train', 'test']), "The task must be train or test."
        self.task = task
        self.object_number = f'{int(object_number):06}'
        self.object_path = self.root_dir / self.task / self.object_number        
        self.image_path = self.object_path / 'rgb'        
        self.cropped_image_path = self.object_path / 'rgb_cropped'        
        self.images = glob.glob(str(self.image_path / '*.png'))        
        self.transform = transform        
        self.COCO_dir = Path('D:\\ImageDataset\\COCO2017\\val2017')
        self.backgrounds = glob.glob(str(self.COCO_dir / '*'))
        self.mask_path = self.object_path / 'mask'
        self.masks = glob.glob(str(self.mask_path / '*'))


        with open(self.object_path / 'scene_gt.json') as json_file:
            gt = json.load(json_file)
        with open(self.object_path / 'scene_gt_info.json') as json_file:
            gt_bbox = json.load(json_file)        
        
        self.num_images = len(gt)
        self.R_matrix = self.gt_R_matrix_load(gt_json=gt)
        self.bbox = self.gt_bbox_load(gt_json=gt_bbox)

        # self.check_cropped_image_and_create(self.images, cropped_image_path=self.cropped_image_path, gt_bbox=gt_bbox)
        # self.cropped_images = glob.glob(str(self.cropped_image_path / '*.png'))
        
        
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = cv2.imread(self.images[idx])
        # image_cropped = cv2.imread(self.cropped_images[idx])
        image_cropped, mask_cropped = self.get_cropped_image_and_mask(image, cv2.imread(self.masks[idx], flags=cv2.IMREAD_GRAYSCALE), bbox=self.bbox[idx])
        pose = self.R_matrix[idx]
        
        image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255).astype(np.float32)
        image_cropped = (cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB) / 255).astype(np.float32)
        # image augmentation sequence
        image_aug = copy.deepcopy(image_cropped)        
        image_aug = self.image_augmentation_scale_and_position(image_aug, mask_cropped, random_background=True)
        # image_aug = self.image_augmentation_random_circle(image_aug)
        # image_aug = self.image_augmentation_blur(image_aug)
        # image_aug = self.image_augmentation_color_change(image_aug)
        
        pose = pose.astype(np.float32)
        sample = {'image': image, 'image_cropped': image_cropped, 'mask_cropped': mask_cropped, 'image_aug': image_aug, 'pose': pose}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_cropped_image_and_mask(self, image, mask, bbox):
        ''' 
        Output image and mask are cropped along the nonzero area and resized into 128*128 shape.
        '''        
        y_min, x_min = bbox[0], bbox[1]
        y_max, x_max = bbox[0]+bbox[2], bbox[1]+bbox[3]
        image_cropped = image[x_min:x_max+1, y_min:y_max+1]
        mask_cropped = mask[x_min:x_max+1, y_min:y_max+1]
        image_cropped = transform.resize(image_cropped, (128,128))
        mask_cropped = transform.resize(mask_cropped, (128,128))
        image_cropped = (np.array(image_cropped) * 255).astype(np.uint8)
        mask_cropped = (mask_cropped > 0.0).astype(np.uint8)

        return image_cropped, mask_cropped

    def image_augmentation_color_change(self, image):
        if np.random.rand(1) > 0.5:
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
        else:
            return image

    def image_augmentation_blur(self, image):
        image_blurred = cv2.GaussianBlur(image, (3,3), cv2.BORDER_DEFAULT)
        return image_blurred

    def image_augmentation_scale_and_position(self, image, mask, random_background=True):
        ''' Scale the image between 0.8 ~ 1.0 random scale factor, and translate the object to random position. '''
        h, w, c = image.shape # [128,128,3]
        scale = (10 - (np.random.rand(1)*2))*0.1
        w_scaled, h_scaled = int(w*scale), int(h*scale)
        w_offset = int(np.random.rand(1)*(w - w_scaled))
        h_offset = int(np.random.rand(1)*(h - h_scaled))
        image_scaled = transform.resize(image, (h_scaled, w_scaled))
        mask_scaled = transform.resize(mask, (h_scaled, w_scaled), anti_aliasing=False)
        mask_idx = (mask_scaled > 0.0)
        if random_background:
            image_bg = cv2.imread(np.random.choice(self.backgrounds))[:,:,::-1]
            image_bg = (image_bg / 255).astype(np.float32)
            h_bg, w_bg, c_bg = image_bg.shape
            h_rand, w_rand  = int(np.random.rand(1)*(h_bg - h)), int(np.random.rand(1)*(w_bg - w))
            image_aug = image_bg[h_rand:h_rand+h, w_rand:w_rand+w, :]
        else:
            image_aug = np.zeros_like(image)
        image_partial = image_aug[h_offset:h_offset+h_scaled, w_offset:w_offset+w_scaled, :] # partial background area
        image_partial[mask_idx] = image_scaled[mask_idx]   # put object foreground on partial background image
        image_aug[h_offset:h_offset+h_scaled, w_offset:w_offset+w_scaled, :] = image_partial # put partial foreground image on whole background image
        return image_aug


    def image_augmentation_random_circle(self, image):
        ''' Put random circles on the object for occlusion. '''
        num_circles = np.random.choice(8,1)[0] # num_circles < 8 (random)
        w, h, c = image.shape
        for _ in range(num_circles):
            x = np.random.choice(w, 1)[0]
            y = np.random.choice(h, 1)[0]
            r = np.random.choice(30, 1)[0]
            color = np.random.rand(3)
            # color = tuple([int(x) for x in color])
            cv2.circle(image, center=(x, y), radius=r, color=color, thickness=-1)

        return image


    def save_cropped_image(self, images, cropped_image_path, gt_bbox):
        ''' 
        Output image is cropped along the nonzero area and resized into 128*128 shape.
        '''
        print(f'Creating cropped {self.task} images. This may take a few seconds.') 

        for i in range(self.num_images):
            image = cv2.imread(images[i])
            # image_idx = np.argwhere((image[:,:,0] != 0) | (image[:,:,1] != 0) | (image[:,:,2] != 0))
            # x_min, y_min = image_idx.min(axis=0)
            # x_max, y_max = image_idx.max(axis=0)
            y_min, x_min = self.bbox[i][0], self.bbox[i][1]
            y_max, x_max = self.bbox[i][0]+self.bbox[i][2], self.bbox[i][1]+self.bbox[i][3]
            image_cropped = image[x_min:x_max+1, y_min:y_max+1]
            image = transform.resize(image_cropped, (128,128))
            image = (np.array(image) * 255).astype(np.uint8)
            cv2.imwrite(str(cropped_image_path / f'{i:06}.png'), image)


    def gt_R_matrix_load(self, gt_json):
        R_matrix = [] # rotation matrix vector in [N,9] shape
        for i in range(len(gt_json)):
            R_matrix.append(np.array(gt_json[str(i)][0]['cam_R_m2c']))        
        return np.array(R_matrix)


    def gt_bbox_load(self, gt_json):
        bbox = [] # bounding box vector in [N,4] shape
        for i in range(len(gt_json)):
            bbox.append(np.array(gt_json[str(i)][0]['bbox_obj']))        
        return np.array(bbox)
         

    def check_cropped_image_and_create(self, images, cropped_image_path, gt_bbox):
        '''
        Creates cropped images in rgb_cropped folder.
        
        images: original images from train or test folder.
        cropped_image_path: path to write cropped images.
        gt_bbox: json file containing ground truth bounding box information.
        '''
        if not os.path.exists(str(cropped_image_path)):
            # if rgb_cropped folder does not exist, make new folder and create cropped images
            os.mkdir(cropped_image_path)
            self.save_cropped_image(images, cropped_image_path, gt_bbox=gt_bbox)
        else:
            # if rgb_cropped folder exists, check the number of files and delete them if the total number is different, then create them again.
            any_files = glob.glob(str(self.cropped_image_path / '*'))
            if self.num_images != len(any_files):
                for f in any_files:
                    os.remove(f)                
                self.save_cropped_image(images, self.cropped_image_path, gt_bbox=gt_bbox)


    def dummy_function(self):
        print('dummy_function called')


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, image_cropped, mask_cropped, image_aug, pose = sample['image'], sample['image_cropped'], sample['mask_cropped'], sample['image_aug'], sample['pose']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image_cropped = image_cropped.transpose((2, 0, 1))
        image_aug = image_aug.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'image_cropped': torch.from_numpy(image_cropped),
                'mask_cropped': mask_cropped,
                'image_aug': torch.from_numpy(image_aug),
                'pose': torch.from_numpy(pose)}