import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    """
    Pretrained ResNet-18
    """
    def __init__(self, feature_dim=512, freeze_layers=True):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        
        
        if freeze_layers:
            for name, param in resnet.named_parameters():
                if not name.startswith("layer4") and not name.startswith("fc"):
                    param.requires_grad = False
        
        # Remove the final classification layer
        # resnet.fc is nn.Linear(in_features=512, out_features=1000)
        in_feats = resnet.fc.in_features
        resnet.fc = nn.Identity()
        
        self.backbone = resnet
        #projecting to desired feature dimension
        self.projector = nn.Sequential(
            nn.Linear(in_feats, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
    
    def forward(self, x):
        # x: (batch_size, 3, H, W)
        features = self.backbone(x)               
        out = self.projector(features)       
        return out
