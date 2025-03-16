import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms


class Read_data(Dataset):
    def __init__(self, root, list_file, transform=None, H=256, W=256, indices=None):
        """
        Args:
            data_path (str): Root directory of the dataset.
            list_file (str): Path to the file (trainval.txt or test.txt) listing image names.
            transform (callable, optional): Optional transform to be applied on a sample.
            H (int): Image height after resizing.
            W (int): Image width after resizing.
            indices (list, optional): Subset indices (for train/val split).
        """
        self.H = H
        self.W = W
        self.data_path = root
        self.transform = transform
        df = pd.read_csv(list_file, sep=" ", header=None)
        names = df[0].values
        self.image_paths = [os.path.join(self.data_path, f"images/{name}.jpg") for name in names]
        self.mask_paths = [os.path.join(self.data_path, f"annotations/trimaps/{name}.png") for name in names]

        # Apply indices for train/val split if provided
        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]
            self.mask_paths = [self.mask_paths[i] for i in indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        try:
            # Read and process the image using PIL (No NumPy)
            image = Image.open(img_path).convert("RGB")  # Convert to RGB
            image = image.resize((self.W, self.H))

            # Read and process the mask using PIL (No NumPy)
            mask = Image.open(mask_path).convert("L")  # Convert to grayscale
            mask = mask.resize((self.W, self.H), resample=Image.NEAREST)

            # Define PyTorch transforms
            transform = transforms.Compose([
                transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
                # transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally
                transforms.RandomRotation(degrees=15)       # Randomly rotate by 1w5 degrees
            ])

            image = transform(image)  # Converts image to (C, H, W)
            mask = transforms.ToTensor()(mask) # Ensures mask shape is (H, W)

            #to bianry mask
            mask = torch.where(mask == 1.0/255.0, 1.0, mask)  # Convert 1 → 0
            mask = torch.where(mask == 3.0/255.0, 1.0, mask)  # Convert 3 → 0
            mask = torch.where(mask == 2.0/255.0, 0.0, mask)  # Convert 2 → 1

            return image, mask
        except Exception as e:
            print(f"Skipping corrupted image: {img_path}, Error: {e}")
            return None, None  # Handle corrupted images

class OxfordPetData:
    def __init__(self, root_dir, batch_size=8, num_workers=8, H=256, W=256):
        """
        Args:
            root_dir (str): Root directory.
            batch_size (int): Batch size for the DataLoader.
            num_workers (int): Number of workers for DataLoader.
        """
        self.root_dir = os.path.join(root_dir,"dataset/oxford-iiit-pet/")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.H = H
        self.W = W
        self._prepare_data()

    def _prepare_data(self):
        trainval_file = os.path.join(self.root_dir, "annotations/trainval.txt")
        test_file = os.path.join(self.root_dir, "annotations/test.txt")
        
        # Create the full train+val dataset using the trainval file
        full_dataset = Read_data(self.root_dir, trainval_file, H=self.H, W=self.W)
        
        # Split indices into train and validation sets
        indices = list(range(len(full_dataset)))
        train_idx, valid_idx = train_test_split(indices, test_size=0.2, random_state=42)

        # Create separate train and validation datasets
        self.train_dataset = Read_data(self.root_dir, trainval_file, H=self.H, W=self.W, indices=train_idx)
        self.valid_dataset = Read_data(self.root_dir, trainval_file, H=self.H, W=self.W, indices=valid_idx)
        self.test_dataset = Read_data(self.root_dir, test_file, H=self.H, W=self.W)

    def get_dataloaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,pin_memory=True)
        valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,pin_memory=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,pin_memory=True)
        return train_loader, valid_loader, test_loader


