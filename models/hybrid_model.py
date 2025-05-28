import torch
import torch.nn as nn
from cnn_feature_extractor import ResNetBackbone

class HybridDeepFakeDetector(nn.Module):
    """
    CNN + LSTM model 
    """
    def __init__(self, input_channels=3, cnn_feature_dim=512, 
                 lstm_hidden_dim=256, num_classes=2):
        super(HybridDeepFakeDetector, self).__init__()
        #CNN
        self.cnn = ResNetBackbone(input_channels, cnn_feature_dim)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len = x.size(0), x.size(1)
        
        #reshape for CNN processing
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        
        # extract features using CNN
        cnn_features = self.cnn(x)  # (batch_size * seq_len, feature_dim)
        
        # Reshaping for lstm
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(cnn_features)
    
        final_features = lstm_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # Classification
        output = self.classifier(final_features)
        
        return output
