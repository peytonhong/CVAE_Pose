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
'''
def load_LM_dataset(PATH):
    dataset_path = Path('D:\ImageDataset\PoseDataset\lm_full')
    partNumber = '000009' # Duck (오리)
    path = dataset_path / 'train' / partNumber
    img_path = path / 'rgb'
    imgs = glob.glob(str(img_path) + '/*.png') # list of rgb images

    with open(path / 'scene_gt.json') as json_file:
        gt = json.load(json_file)
        
    # R_list = []    
    V_list = [] # angular vector in [N x 3 x 1] shape
    for i in range(len(gt)):
        R = np.array(gt[str(i)][0]['cam_R_m2c']).reshape((3,3))    
        # R_list.append(R)
        V_list.append(cv2.Rodrigues(R)[0])
    V_array= np.array(V_list)

    return imgs, V_array
'''
class LineModDataset(Dataset):
    """ Loading LineMod Dataset for Pose Estimation """

    def __init__(self, root_dir, object_number, transform=None):
        """
        Args:
            root_dir (string): Path to the LineMod dataset.
            object_number (int): Unique number of an object. (1, 2, ..., 15) -> converted into str (000001, 000002, ..., 000015)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.object_number = f'{int(object_number):06}'
        self.object_path = self.root_dir / 'train' / self.object_number
        self.image_path = self.object_path / 'rgb'
        self.cropped_image_path = self.object_path / 'rgb_cropped'
        self.images = glob.glob(str(self.image_path / '*.png'))
        self.transform = transform        

        with open(self.object_path / 'scene_gt.json') as json_file:
            gt = json.load(json_file)
        
        self.num_images = len(gt)

        if not os.path.exists(str(self.cropped_image_path)):
            # if rgb_cropped folder does not exist, make new folder and create cropped images
            os.mkdir(self.cropped_image_path)
            self.save_cropped_image()
        else:
            # if rgb_cropped folder exists, check the number of files and delete them if the total number is different, then create them again.
            any_files = glob.glob(str(self.cropped_image_path / '*'))
            if self.num_images != len(any_files):
                for f in any_files:
                    os.remove(f)
                
                self.save_cropped_image()                

        self.cropped_images = glob.glob(str(self.cropped_image_path / '*.png'))

        self.pose_vec = [] # pose vector in [N,3,1] shape
        for i in range(len(gt)):
            R = np.array(gt[str(i)][0]['cam_R_m2c']).reshape((3,3))
            self.pose_vec.append(cv2.Rodrigues(R)[0])
        
        self.pose_vec = np.array(self.pose_vec)
        
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = cv2.imread(self.images[idx])
        image_cropped = cv2.imread(self.cropped_images[idx])
        
        pose = self.pose_vec[idx]

        image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255).astype(np.float32)
        image_cropped = (cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB) / 255).astype(np.float32)
        pose = pose.astype(np.float32)
        sample = {'image': image, 'image_cropped': image_cropped, 'pose': pose}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def save_cropped_image(self):
        ''' Output image is cropped along the nonzero area and resized into 128*128 shape.
        '''
        print('Creating cropped images. This may take a few seconds.')
        for i in range(self.num_images):
            image = cv2.imread(self.images[i])
            image_idx = np.argwhere((image[:,:,0] != 0) | (image[:,:,1] != 0) | (image[:,:,2] != 0))
            x_min, y_min = image_idx.min(axis=0)
            x_max, y_max = image_idx.max(axis=0)
            image_cropped = image[x_min:x_max+1, y_min:y_max+1]
            image = transform.resize(image_cropped, (128,128))
            image = (np.array(image) * 255).astype(np.uint8)
            cv2.imwrite(str(self.cropped_image_path / f'{i:06}.png'), image)

    def dummy_function(self):
        print('dummy_function called')

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, image_cropped, pose = sample['image'], sample['image_cropped'], sample['pose']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image_cropped = image_cropped.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'image_cropped': torch.from_numpy(image_cropped),
                'pose': torch.from_numpy(pose)}