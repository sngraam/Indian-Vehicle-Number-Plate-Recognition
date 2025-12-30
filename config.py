import torch
import os

class Config:
    def __init__(self):
        # Character set for Indian number plates (A-Z, 0-9)
        self.character: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # Image parameters
        self.imgH: int = 32
        self.imgW: int = 100
        self.input_channel: int = 1  # Grayscale

        # Sequence parameters
        self.batch_max_length: int = 10  # Max length for Indian number plates (e.g., MH12AB1234)

        # Model architecture parameters
        self.num_fiducial: int = 20  # Number of fiducial points for TPS
        self.output_channel: int = 512  # ResNet output channels
        self.hidden_dim: int = 256
        self.hidden_size: int = self.hidden_dim  # Alias for model compatibility

        # Training hyperparameters
        self.batch_size: int = 32
        self.epochs: int = 100
        self.learning_rate: float = 0.001
        self.weight_decay: float = 1e-4
        self.grad_clip: float = 5.0  # Gradient clipping
        self.workers: int = 4

        # Device
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset paths (update these to your local paths)
        self.data_dir: str = "./data"  # Directory containing images folder
        self.csv_path: str = "./data/train.csv"  # CSV with annotations
        self.save_dir: str = "./checkpoints"  # Where to save models

        # Calculate num_class: characters + [GO] + [s] tokens
        self.num_class: int = len(self.character) + 2  # +2 for [GO] and [s] tokens

        # Validation split
        self.val_split: float = 0.1

        # Logging
        self.print_interval: int = 100  # Print every N batches
        self.save_interval: int = 5  # Save checkpoint every N epochs


cfg = Config()