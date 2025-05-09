import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
# No logging import needed

class ICLEVERDataset(Dataset):
    def __init__(self, root_dir, json_file, objects_json='objects.json', transform=None):
        """
        Args:
            root_dir (string): Directory with images.
            json_file (string): Path to annotations JSON.
            objects_json (string): Path to object mapping JSON.
            transform (callable, optional): Transform override.
        Simplified error handling: Assumes critical files exist. Uses print.
        """
        self.root_dir = root_dir
        self.filenames = []
        self.object_lists = []
        self.mode = 'train'
        self.object_to_index = {}
        self.num_classes = 0

        # --- Load object mapping --- (No try-except for file/JSON errors)
        # If objects_json doesn't exist or is invalid, this will raise an error
        with open(objects_json, 'r') as f:
            self.object_to_index = json.load(f)
        self.num_classes = len(self.object_to_index)

        # --- Load image annotations --- (No try-except for file/JSON errors)
        # If json_file doesn't exist or is invalid, this will raise an error
        with open(json_file, 'r') as f:
            raw_labels = json.load(f)

        if isinstance(raw_labels, dict): # Assume train format
            self.mode = 'train'
            valid_samples = 0
            for filename, objects in raw_labels.items():
                img_path = os.path.join(self.root_dir, filename)
                # Keep os.path.exists check to filter train data during init
                if os.path.exists(img_path):
                    self.filenames.append(filename)
                    self.object_lists.append(objects)
                    valid_samples += 1
                # else: Optionally print skipped files:
                #     print(f"DEBUG: Train image file not found, skipping: {img_path}")
            if valid_samples == 0 and len(raw_labels) > 0 :
                 print(f"WARNING: Loaded 0 samples but raw train data had > 0 entries. Check image paths in {self.root_dir}.")

        elif isinstance(raw_labels, list): # Assume test/val format
            self.mode = 'test'
            self.object_lists = raw_labels
            self.filenames = [f"test_{i:05d}" for i in range(len(raw_labels))] # Dummy names
        else:
            # Keep check for structure error
            raise ValueError(f"Unsupported JSON structure in {json_file}. Expected dict or list.")



        # --- Define transforms ---
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Correct normalization
            ])
        else:
            self.transform = transform


    def __len__(self):
        return len(self.object_lists)

    def __getitem__(self, idx):
        objects = self.object_lists[idx]

        # --- Create Multi-Hot Encoded Condition Vector ---
        condition_vector = torch.zeros(self.num_classes, dtype=torch.float)
        unknown_objects = []
        for obj_str in objects:
            if obj_str in self.object_to_index:
                obj_idx = self.object_to_index[obj_str]
                condition_vector[obj_idx] = 1.0
            else:
                 unknown_objects.append(obj_str)
        # Optional: Print warnings for unknown objects (can be verbose)
        if unknown_objects:
             print(f"WARNING: Sample {idx}: Found unknown objects: {unknown_objects}")


        # --- Load Image (Only for Train Mode) ---
        if self.mode == 'train':
            img_name = self.filenames[idx]
            img_path = os.path.join(self.root_dir, img_name)
            # Keep minimal try-except around file IO during item loading for robustness
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, condition_vector
        else: # Test mode
            # Return condition vector and None for image
            return condition_vector
