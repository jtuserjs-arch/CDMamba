"""
CD Dataset for Hi-CNA dataset
"""
import os
from PIL import Image
import numpy as np
import data.util as Util
from torch.utils.data import Dataset
import torchvision
import torch

totensor = torchvision.transforms.ToTensor()

"""
Hi-CNA Dataset Structure
├─image1 (A images - tif)
├─image2 (B images - tif) 
├─change (label - png)
"""

IMG_FOLDER_NAME = 'image1'  # A images
IMG_POST_FOLDER_NAME = 'image2'  # B images
LABEL_FOLDER_NAME = 'change'  # label

# 获取各个文件夹的路径
def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name + '.png')

def get_img_post_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name + '.png')

def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, LABEL_FOLDER_NAME, img_name + '.png')

# 直接从目录获取图像名称列表
def get_img_name_list_from_dir(image_dir):
    img_names = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            # 移除扩展名，只保留基本文件名
            base_name = os.path.splitext(filename)[0]
            img_names.append(base_name)
    return np.array(img_names)

class HIDataset(Dataset):
    def __init__(self, root_dir, resolution=512, split='train', data_len=-1, label_transform=None):

        self.root_dir = os.path.join(root_dir,split)
        self.resolution = resolution
        self.data_len = data_len
        self.split = split  # train / val / test
        self.label_transform = label_transform

        # 直接从image1目录获取图像名称列表
        image_dir = os.path.join(self.root_dir, IMG_FOLDER_NAME)
        self.img_name_list = get_img_name_list_from_dir(image_dir)

        self.dataset_len = len(self.img_name_list)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.dataset_len, self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # 获取基础文件名
        base_name = self.img_name_list[index % self.data_len]

        A_path = get_img_path(self.root_dir, base_name)
        B_path = get_img_post_path(self.root_dir, base_name)
        # 直接读取 PNG，不进行任何灰度扩展或颜色转换
        img_A = Image.open(A_path)
        img_B = Image.open(B_path)

        L_path = get_label_path(self.root_dir, base_name)
        img_label = Image.open(L_path)

        img_label = np.array(img_label, dtype=np.uint8)
        img_A = np.array(img_A, dtype=np.uint8)
        img_B = np.array(img_B, dtype=np.uint8)


        img_label = torch.from_numpy(img_label)
        img_A = Util.transform_augment_cd(img_A, min_max=(-1, 1))
        img_B = Util.transform_augment_cd(img_B, min_max=(-1, 1))

        if img_label.dim() > 2:
            img_label = img_label[0]

        return {'A': img_A, 'B': img_B, 'L': img_label, 'Index': index, 'name': base_name}


if __name__ == '__main__':
    # 测试数据集
    root_dir = r'/path/to/your/Hi-CNA/dataset'
    dataset = HIDataset(root_dir=root_dir, split='train')

    print(f"Dataset length: {len(dataset)}")
    print(f"Image names: {dataset.img_name_list[:5]}")  # 显示前5个图像名称

    # 测试一个样本
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"A shape: {sample['A'].shape}")
    print(f"B shape: {sample['B'].shape}")
    print(f"L shape: {sample['L'].shape}")
    print(f"Name: {sample['name']}")