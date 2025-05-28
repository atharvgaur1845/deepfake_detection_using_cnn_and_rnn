import torch
import torch.nn as nn
from cnn_feature_extractor import ResNetBackbone

class HybridDeepFakeDetector(nn.Module):
    def __init__(self, input_channels=3, cnn_feature_dim=512, 
                 lstm_hidden_dim=256, num_classes=2):
        super(HybridDeepFakeDetector, self).__init__()
        

        self.cnn = ResNetBackbone(feature_dim=cnn_feature_dim)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        #classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len = x.size(0), x.size(1)

        
        #(batch_size * seq_len, channels, height, width)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        print(f"Reshaped for CNN: {x.shape}")
        
        # Extract features using CNN
        cnn_features = self.cnn(x)  # Should be (batch_size * seq_len, feature_dim)
        print(f"CNN features shape: {cnn_features.shape}")
        
        # Reshape for LSTM: (batch_size, seq_len, feature_dim)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        print(f"Reshaped for LSTM: {cnn_features.shape}")
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_features)
        
        
        final_features = lstm_out[:, -1, :]
        output = self.classifier(final_features)
        return output
