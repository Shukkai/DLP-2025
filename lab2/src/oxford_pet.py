import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from urllib.request import urlretrieve
import shutil
seed = 42

# Set the Python built-in random module seed
random.seed(seed)

def joint_transform(image, mask, degrees=15, flip_prob=0.5):
    """
    Applies a random rotation to both the image and mask, then converts them to tensors.
    
    Args:
        image (PIL.Image): The input image.
        mask (PIL.Image): The corresponding segmentation mask.
        degrees (float): Maximum absolute rotation angle (in degrees). The angle will be chosen uniformly from [-degrees, degrees].
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The rotated image and mask as tensors.
    """
    # Choose a random rotation angle between -degrees and +degrees
    angle = random.uniform(-degrees, degrees)
    
    # Rotate both the image and the mask using the same angle
    rotated_image = TF.rotate(image, angle)
    rotated_mask = TF.rotate(mask, angle)

    # Randomly perform a horizontal flip with probability flip_prob.
    if random.random() < flip_prob:
        rotated_image = TF.hflip(rotated_image)
        rotated_mask = TF.hflip(rotated_mask)
    
    # Convert the rotated PIL images to tensors
    image_tensor = TF.to_tensor(rotated_image)
    mask_tensor = TF.to_tensor(rotated_mask)
    
    return image_tensor, mask_tensor

class Read_data(Dataset):
    def __init__(self, root, list_file, H=256, W=256, indices=None, is_test = False):
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
        if not os.path.exists(self.data_path):
            print(f"Directory '{self.data_path}' does not exist. Creating it now...")
            os.makedirs(self.data_path)
        self.download(self.data_path)
        df = pd.read_csv(list_file, sep=" ", header=None)
        names = df[0].values
        self.image_paths = [os.path.join(self.data_path, f"images/{name}.jpg") for name in names]
        self.mask_paths = [os.path.join(self.data_path, f"annotations/trimaps/{name}.png") for name in names]
        self.is_test = is_test

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
            # Read and process the image using PIL
            image = Image.open(img_path).convert("RGB")  # Convert to RGB
            image = image.resize((self.W, self.H))

            # Read and process the mask using PIL
            mask = Image.open(mask_path).convert("L")  # Convert to grayscale
            mask = mask.resize((self.W, self.H), resample=Image.NEAREST)

            # Define PyTorch transforms
            if self.is_test == False:
                image, mask = joint_transform(image=image, mask=mask)
            else:
                
                image = TF.to_tensor(image)  # Converts image to (C, H, W)
                mask = TF.to_tensor(mask) # Ensures mask shape is (H, W)
            #to bianry mask
            mask = torch.where(mask == 1.0/255.0, 1.0, mask)  # Convert 1 → 0
            mask = torch.where(mask == 3.0/255.0, 1.0, mask)  # Convert 3 → 0
            mask = torch.where(mask == 2.0/255.0, 0.0, mask)  # Convert 2 → 1
            return image, mask

        except Exception as e:
            print(f"Skipping corrupted image: {img_path}, Error: {e}")
            return None, None  # Handle corrupted images

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)
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
        full_dataset.download(self.root_dir)
        
        # Split indices into train and validation sets
        indices = list(range(len(full_dataset)))
        train_idx, valid_idx = train_test_split(indices, test_size=0.2, random_state=42)

        # Create separate train and validation datasets
        self.train_dataset = Read_data(self.root_dir, trainval_file, H=self.H, W=self.W, indices=train_idx, is_test=False)
        self.valid_dataset = Read_data(self.root_dir, trainval_file, H=self.H, W=self.W, indices=valid_idx, is_test=False)
        self.test_dataset = Read_data(self.root_dir, test_file, H=self.H, W=self.W,is_test=True)

    def get_dataloaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,pin_memory=True)
        valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,pin_memory=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,pin_memory=True)
        return train_loader, valid_loader, test_loader

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)


