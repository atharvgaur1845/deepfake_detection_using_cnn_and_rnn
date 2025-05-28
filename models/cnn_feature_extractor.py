import torch
import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    """
    CNN for extracting features
    """
    def __init__(self, input_channels=3, feature_dim=512):
        super(CNNFeatureExtractor, self).__init__()
        
        self.features = nn.Sequential(

            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # adaptive pooling 
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        #fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        features = self.features(x)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features
