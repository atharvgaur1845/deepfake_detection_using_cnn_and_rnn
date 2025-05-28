import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    def __init__(self, feature_dim=512, freeze_layers=True):
        super(ResNetBackbone, self).__init__()
        
        # Load pretrained ResNet-18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if freeze_layers:
            for name, param in resnet.named_parameters():
                if not name.startswith("layer4") and not name.startswith("fc"):
                    param.requires_grad = False
        
        modules = list(resnet.children())[:-1]  # Remove last FC layer
        self.backbone = nn.Sequential(*modules)
        
        # Add adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Project to desired feature dimension
        self.projector = nn.Sequential(
            nn.Linear(512, feature_dim),  # ResNet18 outputs 512 features
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        # x: (batch_size, 3, H, W)
        features = self.backbone(x)           # (batch_size, 512, H', W')
        features = self.adaptive_pool(features)  # (batch_size, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size, 512)
        features = self.projector(features)   # (batch_size, feature_dim)
        return features
