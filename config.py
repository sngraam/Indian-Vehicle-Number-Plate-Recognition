import torch
class Config:
    def __init__(self):
        self.character: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # Training parameters
        self.batch_max_length: int = 4
        self.workers: int = 8
        self.imgH: int = 32
        self.imgW: int = 100

        # Model parameters
        self.hidden_dim: int = 256
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_fiducial = 20
        self.input_channel = 1

        # Attributes required by the Model class (based on argparse in models.py)
        self.Transformation: str = "None"
        self.FeatureExtraction: str = "ResNet"
        self.SequenceModeling: str = "BiLSTM"
        self.Prediction: str = "CTC"
        self.output_channel: int = 512 # Default value from models.py argparse
        self.hidden_size: int = self.hidden_dim # Model expects hidden_size, align with hidden_dim

        # Calculate num_class based on the character set
        self.num_class = len(self.character)


cfg = Config()