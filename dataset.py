"""
Dataset class for Indian Vehicle Number Plate OCR
Handles loading images, cropping bounding boxes, and encoding labels
"""

import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageEnhance
import pandas as pd
import numpy as np


class NumberPlateDataset(Dataset):
    """
    Dataset for loading number plate images with bounding box annotations.
    
    Expected CSV format:
        filename,width,height,label,xmin,ymin,xmax,ymax
        img_00000.jpg,272,306,HP54C6564,105,183,167,197
    """
    
    def __init__(self, csv_path, img_dir, converter, cfg, transform=None, augment=False):
        """
        Args:
            csv_path: Path to CSV file with annotations
            img_dir: Directory containing images
            converter: AttnLabelConverter instance for encoding labels
            cfg: Config object with imgH, imgW, batch_max_length
            transform: Optional transforms to apply
            augment: Whether to apply data augmentation
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.converter = converter
        self.cfg = cfg
        self.transform = transform
        self.augment = augment
        
        # Filter out labels that are too long
        self.df = self.df[self.df['label'].str.len() <= cfg.batch_max_length].reset_index(drop=True)
        
        # Filter labels to only contain valid characters
        valid_chars = set(cfg.character)
        def is_valid_label(label):
            return all(c in valid_chars for c in str(label).upper())
        
        self.df = self.df[self.df['label'].apply(is_valid_label)].reset_index(drop=True)
        
        print(f"Dataset loaded: {len(self.df)} samples after filtering (augment={augment})")
    
    def __len__(self):
        return len(self.df)
    
    def apply_augmentation(self, image):
        """Apply random augmentations to the image."""
        # Random rotation (-5 to 5 degrees)
        if random.random() < 0.3:
            angle = random.uniform(-5, 5)
            image = image.rotate(angle, fillcolor=128)
        
        # Random brightness
        if random.random() < 0.3:
            factor = random.uniform(0.7, 1.3)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)
        
        # Random contrast
        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)
        
        # Random blur
        if random.random() < 0.1:
            image = image.filter(ImageFilter.GaussianBlur(radius=1))
        
        return image
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        
        # Crop bounding box region
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # Ensure valid bbox coordinates
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        
        # Add small padding if bbox is too small
        if xmax - xmin < 10:
            xmax = xmin + 10
        if ymax - ymin < 5:
            ymax = ymin + 5
            
        # Crop and resize
        cropped = image.crop((xmin, ymin, xmax, ymax))
        
        # Convert to grayscale
        cropped = cropped.convert('L')
        
        # Resize to model input size
        cropped = cropped.resize((self.cfg.imgW, self.cfg.imgH), Image.BILINEAR)
        
        # Apply augmentation if enabled
        if self.augment:
            cropped = self.apply_augmentation(cropped)
        
        # Apply transforms if any
        if self.transform:
            cropped = self.transform(cropped)
        else:
            # Default: convert to tensor and normalize
            cropped = np.array(cropped, dtype=np.float32)
            cropped = (cropped / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]
            cropped = torch.FloatTensor(cropped).unsqueeze(0)  # [1, H, W]
        
        # Get label (uppercase for consistency)
        label = str(row['label']).upper()
        
        return cropped, label


def collate_fn(batch, converter, cfg):
    """
    Custom collate function to batch images and encode labels.
    
    Args:
        batch: List of (image, label) tuples
        converter: AttnLabelConverter instance
        cfg: Config object
    
    Returns:
        images: [B, 1, H, W] tensor
        text: [B, max_length+2] tensor (encoded labels with [GO] and [s])
        length: [B] tensor (actual lengths)
    """
    images, labels = zip(*batch)
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    # Encode labels
    text, length = converter.encode(labels, batch_max_length=cfg.batch_max_length)
    
    return images, text, length


class AlignCollate:
    """Collate function wrapper for DataLoader"""
    
    def __init__(self, converter, cfg):
        self.converter = converter
        self.cfg = cfg
    
    def __call__(self, batch):
        return collate_fn(batch, self.converter, self.cfg)


def create_dataloaders(cfg, converter):
    """
    Create train and validation dataloaders.
    
    Args:
        cfg: Config object with data paths and training params
        converter: AttnLabelConverter instance
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    
    img_dir = os.path.join(cfg.data_dir, 'images')
    
    # Create train dataset with augmentation
    train_dataset = NumberPlateDataset(
        csv_path=cfg.csv_path,
        img_dir=img_dir,
        converter=converter,
        cfg=cfg,
        augment=True  # Enable augmentation for training
    )
    
    # Create validation dataset (no augmentation)
    val_dataset = NumberPlateDataset(
        csv_path=cfg.csv_path,
        img_dir=img_dir,
        converter=converter,
        cfg=cfg,
        augment=False
    )
    
    # Split indices for train/val
    total_size = len(train_dataset)
    indices = list(range(total_size))
    random.seed(42)
    random.shuffle(indices)
    
    val_size = int(total_size * cfg.val_split)
    train_size = total_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # Create collate function
    collate = AlignCollate(converter, cfg)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader

