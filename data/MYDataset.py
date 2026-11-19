import os
import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import data.util as Util


class MYDataset(Dataset):
    def __init__(self, root_dir, split="train", data_len=-1, label_transform=None):
        """
        :param root_dir: 数据集根目录 (包含 x1/ x2/ y_out/ list/)
        :param split: 'train' 或 'val'
        :param data_len: 限制数据长度，-1 表示使用全部
        :param label_transform: 标签的额外 transform
        """
        self.root_dir = root_dir
        self.split = split
        self.label_transform = label_transform

        # CSV 文件路径
        csv_path = os.path.join(root_dir, "list", f"{split}.csv")
        assert os.path.exists(csv_path), f"CSV 文件不存在: {csv_path}"

        # 读取 csv 文件
        self.samples = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "A": os.path.join(root_dir, row["x1_path"]),
                    "B": os.path.join(root_dir, row["x2_path"]),
                    "L": os.path.join(root_dir, row["y_path"])
                })

        self.dataset_len = len(self.samples)
        if data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.dataset_len, data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        sample = self.samples[index % self.data_len]

        A_path = os.path.normpath(sample["A"])
        B_path = os.path.normpath(sample["B"])
        L_path = os.path.normpath(sample["L"])

        # 读取 A, B 图像
        img_A = np.array(Image.open(A_path), dtype=np.uint8)
        img_B = np.array(Image.open(B_path), dtype=np.uint8)

        # 读取 label
        img_label = np.array(Image.open(L_path), dtype=np.uint8)

        # 转 tensor
        img_A = Util.transform_augment_cd(img_A, min_max=(-1, 1))
        img_B = Util.transform_augment_cd(img_B, min_max=(-1, 1))
        img_label = torch.from_numpy(img_label)

        if img_label.dim() > 2:
            img_label = img_label[0]

        base_name = os.path.splitext(os.path.basename(sample["A"]))[0]

        return {"A": img_A, "B": img_B, "L": img_label, "Index": index, "name": base_name}


if __name__ == "__main__":
    root_dir = "/path/to/dataset"
    dataset = CDDataset(root_dir=root_dir, split="train")

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"A shape: {sample['A'].shape}")
    print(f"B shape: {sample['B'].shape}")
    print(f"L shape: {sample['L'].shape}")
    print(f"Name: {sample['name']}")
