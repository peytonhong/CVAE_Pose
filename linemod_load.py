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
from tqdm import tqdm
from plyfile import PlyData
from rendering.renderer_xyz import Renderer
from rendering.model import Model3D
from sys import platform
import time

class LineModDataset(Dataset):
    """ Loading LineMod Dataset for Pose Estimation """

    def __init__(self, root_dir, background_dir, task, object_number, transform=None, augmentation=False, rendering=False, use_offline_data=True, use_useful_data=True):
        """
        Args:
            root_dir (string): Path to the LineMod dataset.
            object_number (int): Unique number of an object. (1, 2, ..., 15) -> converted into str (000001, 000002, ..., 000015)
            transform (callable, optional): Optional transform to be applied on a sample.
            use_offline_data: use saved(offline) images, if False: create images on every iterations (steps)
        """
        self.root_dir = Path(root_dir)
        assert (task in ['train', 'test']), "The task must be train or test."
        self.task = task
        self.object_number = f'{int(object_number):06}'
        self.object_path = self.root_dir / self.task / self.object_number        
        self.image_path = self.object_path / 'rgb'        
        self.images = sorted(glob.glob(str(self.image_path / '*.png')))
        self.transform = transform        
        self.COCO_dir = background_dir
        self.backgrounds = glob.glob(str(self.COCO_dir / '*'))
        self.mask_path = self.object_path / 'mask'
        self.masks = sorted(glob.glob(str(self.mask_path / '*')))
        self.augmentation = augmentation
        self.rendering = rendering
        self.use_offline_data = use_offline_data
        self.use_useful_data = use_useful_data
        self.images_useful = []

        # image augmentation paths (To reduce training time by saving augmented images in advance)
        self.max_num_aug_images = 20000
        self.cropped_image_path = self.object_path / 'rgb_cropped'
        self.aug_image_path = self.object_path / 'rgb_aug'           
        self.aug_useful_path = self.object_path / 'rgb_useful'
        self.cropped_mask_path = self.object_path / 'mask_cropped'
        self.gt_image_path = self.object_path / 'rbg_gt_cropped'
        if not self.cropped_image_path.exists():
            self.cropped_image_path.mkdir()
        if not self.aug_image_path.exists():
            self.aug_image_path.mkdir()
        if not self.aug_useful_path.exists():
            self.aug_useful_path.mkdir()
        if not self.cropped_mask_path.exists():
            self.cropped_mask_path.mkdir()
        if not self.gt_image_path.exists():
            self.gt_image_path.mkdir()
        
        with open(self.object_path / 'scene_gt.json') as json_file:
            gt = json.load(json_file)
        with open(self.object_path / 'scene_gt_info.json') as json_file:
            gt_bbox = json.load(json_file) 
        self.R_matrix = self.gt_R_matrix_load(gt_json=gt)
        self.bbox = self.gt_bbox_load(gt_json=gt_bbox)

        # Check number of saved images and recreate if the number is not matched to self.max_num_aug_images
        if self.use_offline_data:
            self.save_sample_images()
            self.images_cropped = sorted(self.cropped_image_path.glob('*'))
            self.masks_cropped = sorted(self.cropped_mask_path.glob('*'))
            self.images_aug = sorted(self.aug_image_path.glob('*'))
            self.images_gt_cropped = sorted(self.gt_image_path.glob('*'))        

        # pointcloud model data import
        if self.rendering:
            model_path = str(self.root_dir / 'models' / f'obj_{object_number:06d}.ply')
            if platform == 'win32': # only for Windows
                model_path = model_path.replace(os.sep, os.altsep) # replace '\' to '/'
            self.obj_model = Model3D()
            self.obj_model.load(model_path, scale=0.001)        
            # model_data = PlyData.read(model_path)        
            # (self.model_x, self.model_y, self.model_z) = (np.array(model_data['vertex'][t]) for t in ('x', 'y', 'z'))
            # (r, g, b, a) = (np.array(model_data['vertex'][t])/255 for t in ('red', 'green', 'blue', 'alpha'))
            # self.model_colors = np.vstack((r,g,b,a)).transpose()
            scene_camera_path = self.root_dir / 'train' / f'{object_number:06d}' / 'scene_camera.json'
            with open(scene_camera_path) as json_file:
                scene_camera = json.load(json_file)
            cam_K = scene_camera['0']['cam_K']
            cam_K = np.array(cam_K).reshape((3,3))
            cam_T = gt['0'][0]['cam_t_m2c']
            self.cam_T = np.array(cam_T) / 1000 # [mm] -> [m]        
            self.ren = Renderer((640, 480), cam_K) # shape: (width, height)
        
    def __len__(self):
        if self.use_offline_data:
            return len(self.images_aug)
        else:
            return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.use_offline_data:
            image_aug = cv2.imread(str(self.images_aug[idx]))
            label = int(self.images_aug[idx].stem[:6])
            image_cropped = cv2.imread(str(self.cropped_image_path / f'{label:06}.png'))
            mask_cropped = cv2.imread(str(self.cropped_mask_path / f'{label:06}.png'))
            image_gt_cropped = cv2.imread(str(self.gt_image_path / f'{label:06}.png'))
            image = cv2.imread(self.images[label])
            pose = self.R_matrix[label].astype(np.float32)
            
            # image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255).astype(np.float32)
            # image_aug = (cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB) / 255).astype(np.float32)
            # image_cropped = (cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB) / 255).astype(np.float32)
            # image_gt_cropped = (cv2.cvtColor(image_gt_cropped, cv2.COLOR_BGR2RGB) / 255).astype(np.float32)
            # mask_cropped = (mask_cropped / 255).astype(np.float32)

            # sample = {'image': image, 'image_cropped': image_cropped, 'mask_cropped': mask_cropped, 'image_aug': image_aug, 'image_gt_cropped': image_gt_cropped, 'pose': pose}
        else:
            image_path = self.images[idx]
            sample = self.load_sample(image_path)
            image = sample['image']
            image_cropped = sample['image_cropped']
            mask_cropped = sample['mask_cropped']
            image_aug = sample['image_aug']
            image_gt_cropped = sample['image_gt_cropped']
            pose = sample['pose']

        image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255).astype(np.float32)
        image_aug = (cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB) / 255).astype(np.float32)
        image_cropped = (cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB) / 255).astype(np.float32)
        image_gt_cropped = (cv2.cvtColor(image_gt_cropped, cv2.COLOR_BGR2RGB) / 255).astype(np.float32)
        # mask_cropped = (mask_cropped / 255).astype(np.float32)

        sample = {'image': image, 'image_cropped': image_cropped, 'mask_cropped': mask_cropped, 'image_aug': image_aug, 'image_gt_cropped': image_gt_cropped, 'pose': pose}
        if self.use_useful_data:
            self.images_useful.append({'image_aug':image_aug, 'label': int(Path(self.images[idx]).stem[:6])})

        if self.transform:
            sample = self.transform(sample)
        return sample

    def save_useful_images(self):
        print(f'Saving useful {len(self.images_useful)} images due to largely decreased test R loss.')
        current_time = str(int(time.time()))
        for sample in self.images_useful:
            image_aug = (cv2.cvtColor(sample['image_aug'], cv2.COLOR_BGR2RGB)*255).astype(np.uint8)
            label = sample['label']            
            cv2.imwrite((str(self.aug_useful_path / Path(f'{label:06d}')) + '_' + current_time + '.png'), image_aug)
        self.images_useful = []
    
    def remove_useful_images(self):
        # delete all existing useful images
        print(f'Remove existing useful images.')
        files = glob.glob(str(self.aug_useful_path) + '/*')
        for f in files:
            os.remove(f)


    def save_sample_images(self):
        num_aug_images = len(sorted(self.aug_image_path.glob('*')))
        if self.task == 'train':            
            is_same = (num_aug_images == self.max_num_aug_images)
        else:
            is_same = (num_aug_images == len(self.images))

        if not is_same: # if the number of augmented images is not matched.
            print(f'Creating new augmented data for {self.task} images.')
            cropped_image_path = np.array(sorted(self.cropped_image_path.glob('*')))
            aug_image_path = np.array(sorted(self.aug_image_path.glob('*')))
            total_image_path = np.hstack((cropped_image_path, aug_image_path))
            for path in total_image_path:
                path.unlink()
        
            images_to_save_list = np.array(self.images)
            images_random_choice = np.random.choice(self.images, size=(self.max_num_aug_images - len(self.images)))            
            images_to_save_list = np.hstack((images_to_save_list, images_random_choice))
            cnt = 0
            saved_image_path = []
            for image_path in tqdm(images_to_save_list):                
                sample = self.load_sample(image_path)
                if not image_path in saved_image_path:                    
                    cv2.imwrite(str(self.cropped_image_path / Path(str(Path(image_path).stem) + '.png')), sample['image_cropped'])
                    cv2.imwrite(str(self.gt_image_path / Path(str(Path(image_path).stem) + '.png')), sample['image_gt_cropped'])
                    cv2.imwrite(str(self.cropped_mask_path / Path(str(Path(image_path).stem) + '.png')), sample['mask_cropped'])
                    if self.task == 'test':
                        cv2.imwrite(str(self.aug_image_path / Path(str(Path(image_path).stem) + '.png')), sample['image_aug']) # image_aug which is same with image_cropped
                    saved_image_path.append(image_path)
                else:
                    cnt += 1
                if self.task == 'train':
                    cv2.imwrite(str(self.aug_image_path / Path(str(Path(image_path).stem) + '_' + f'{cnt:05}.png')), sample['image_aug'])
                    

    def load_sample(self, image_path):
        image = cv2.imread(image_path)
        idx = int(Path(image_path).stem) # stem: The final path component, without its suffix
        image_cropped, image_gt_cropped, mask_cropped = self.get_cropped_image_and_mask(image, cv2.imread(self.masks[idx], flags=cv2.IMREAD_GRAYSCALE), bbox=self.bbox[idx])
                    
        if self.augmentation:
            # image augmentation sequence
            image_aug = copy.deepcopy(image_cropped)        
            image_aug = self.image_augmentation_color_change(image_aug) # gamma correction
            image_aug = self.image_augmentation_scale_and_position(image_aug, mask_cropped, random_background=True)
            image_aug = self.image_augmentation_random_circle(copy.deepcopy(image_aug))
            # image_aug = self.image_augmentation_blur(image_aug)            
        else:
            image_aug = image_cropped
        pose = self.R_matrix[idx].astype(np.float32)

        sample = {'image': image, 'image_cropped': image_cropped, 'mask_cropped': mask_cropped, 'image_aug': image_aug, 'image_gt_cropped': image_gt_cropped, 'pose': pose} 
        
        return sample

    def get_cropped_image_and_mask(self, image, mask, bbox):
        ''' 
        Output image and mask are cropped along the nonzero area and resized into 128*128 shape.
        '''        
        x_min, y_min = bbox[0], bbox[1]
        x_max, y_max = bbox[0]+bbox[2], bbox[1]+bbox[3]
        # Mask area to be square (offset x or y points)
        x_center = int((x_max + x_min)/2)
        y_center = int((y_max + y_min)/2)
        if (x_center - x_min) > (y_center - y_min):
            y_offset = (x_center - x_min) - (y_center - y_min)
            y_min -= y_offset
            y_max += y_offset
        elif (x_center - x_min) < (y_center - y_min):
            x_offset = (y_center - y_min) - (x_center - x_min)
            x_min -= x_offset
            x_max += x_offset
        
        image_cropped = image[y_min:y_max+1, x_min:x_max+1]
        mask_cropped = mask[y_min:y_max+1, x_min:x_max+1]

        image_gt_cropped = cv2.bitwise_and(image_cropped, image_cropped, mask=mask_cropped)

        image_cropped = cv2.resize(image_cropped, (128,128), interpolation=cv2.INTER_LINEAR )
        mask_cropped = cv2.resize(mask_cropped, (128,128), interpolation=cv2.INTER_LINEAR )
        image_gt_cropped = cv2.resize(image_gt_cropped, (128,128), interpolation=cv2.INTER_LINEAR )

        mask_cropped = (mask_cropped == 255).astype(np.uint8)*255
        
        return image_cropped, image_gt_cropped, mask_cropped
        
    def adjust_gamma(self, image, gamma=1.0):        
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)


    def image_augmentation_color_change(self, image):
        ''' color change from gamma correction '''
        gamma = np.random.rand(1) + 0.5 # gamma range: 0.5~1.5
        gamma_corrected_image = self.adjust_gamma(image, gamma=gamma)
        return gamma_corrected_image

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
        image_scaled = cv2.resize(image, (h_scaled, w_scaled), interpolation=cv2.INTER_LINEAR )
        mask_scaled = cv2.resize(mask, (h_scaled, w_scaled), interpolation=cv2.INTER_LINEAR )
        mask_idx = (mask_scaled == 255)
        if random_background:
            image_bg = cv2.imread(np.random.choice(self.backgrounds))[:,:,::-1]
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
        num_circles = np.random.choice(10,1)[0] # num_circles < 10 (random)
        h, w, c = image.shape
        for _ in range(num_circles):
            x = np.random.choice(h, 1)[0]
            y = np.random.choice(w, 1)[0]
            r = np.random.choice(10, 1)[0]
            color = np.random.rand(3)*255
            cv2.circle(image, center=(x, y), radius=r, color=color, thickness=-1)

        return image


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
         




class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, image_cropped, mask_cropped, image_aug, image_gt_cropped, pose = sample['image'], sample['image_cropped'], sample['mask_cropped'], sample['image_aug'], sample['image_gt_cropped'], sample['pose']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image_cropped = image_cropped.transpose((2, 0, 1))
        image_aug = image_aug.transpose((2, 0, 1))
        image_gt_cropped = image_gt_cropped.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'image_cropped': torch.from_numpy(image_cropped),
                'mask_cropped': mask_cropped,
                'image_aug': torch.from_numpy(image_aug),
                'image_gt_cropped': image_gt_cropped,
                'pose': torch.from_numpy(pose)}