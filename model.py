import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


class Model(nn.Module):

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg  # Store config for forward pass
        
        """ Transformation """
        
        self.transformation = TPS_SpatialTransformerNetwork(
            F=cfg.num_fiducial, I_size=(cfg.imgH, cfg.imgW), I_r_size=(cfg.imgH, cfg.imgW), I_channel_num=cfg.input_channel)

        self.FeatureExtraction = ResNet_FeatureExtractor(cfg.input_channel, cfg.output_channel)
        self.FeatureExtraction_output = cfg.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1


        self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, cfg.hidden_size, cfg.hidden_size),
                BidirectionalLSTM(cfg.hidden_size, cfg.hidden_size, cfg.hidden_size))

        self.SequenceModeling_output = cfg.hidden_size
        self.Prediction = Attention(self.SequenceModeling_output, cfg.hidden_size, cfg.num_class)

    def forward(self, input, text, is_train=True):
        """ Transformation stage """

        input = self.transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)
        prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.cfg.batch_max_length)

        return prediction