# DeepFake Detection Project Documentation

## 1. Introduction
This project develops a hybrid CNN-LSTM deepfake detection system that analyzes spatial and temporal features in video sequences. The system combines ResNet-18 for spatial feature extraction with bidirectional LSTM for temporal pattern recognition, achieving 87.69% accuracy on test data. Key learnings include the critical importance of temporal modeling and robust preprocessing pipelines for effective deepfake detection.

## 2. Thought Process
After reviewing literature on deepfake detection methods including XceptionNet (91.5% accuracy) and GenConViT (93.8%), I selected a hybrid approach balancing performance and computational efficiency. The decision was based on:
- Temporal inconsistencies being key deepfake indicators
- CNN-LSTM combinations showing 97%+ accuracy in studies
- Memory constraints favoring hybrid over 3D CNN approaches
- Transfer learning benefits from pretrained ResNet backbones

## 3. Blockers
Major challenges encountered:
- **GPU Memory Limitations**: CUDA out of memory errors requiring batch size reduction to 2 and sequence length optimization
- **Training Instability**: Initial accuracy stuck at 52% due to class imbalance and inappropriate learning rates
- **Data Pipeline Complexity**: Face detection failures and import path issues during modular development

## 4. Approach

### Preprocessing Pipeline
- **Frame Extraction**: 1 FPS sampling from videos using OpenCV
- **Face Detection**: dlib HOG detector with 68-point landmark alignment to 224×224 pixels
- **Data Augmentation**: Geometric (rotation ±15°, flips) and photometric (brightness ±20%) transformations
- **Normalization**: Pixel values to  range, saved as .npy files for efficient loading

### Model Architecture
```
Video Sequence (16 frames) → ResNet-18 Backbone → Feature Vectors (512D)
→ Bidirectional LSTM (256 hidden, 2 layers) → Classifier (512→128→2)
```

### Training Configuration
- **Loss**: Cross-entropy with class weighting [0.48, 0.52] for balance
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Regularization**: Dropout (0.3 LSTM, 0.5 classifier), gradient clipping (max_norm=1.0)

## 5. Comparative Study

| Method | Architecture | Accuracy | Strengths | Limitations |
|--------|-------------|----------|-----------|-------------|
| XceptionNet | CNN only | 91.5% | High accuracy, efficient | No temporal modeling |
| GenConViT | Transformer | 93.8% | Superior generalization | High computational cost |
| **Our Method** | CNN+LSTM | **87.7%** | Temporal awareness, balanced | Memory constraints |

## 6. Results

### Performance Metrics
- **Overall Accuracy**: 87.69%
- **AUC-ROC**: ~0.90 (estimated from classification performance)
- **Dataset**: 2,502 sequences (1,308 Real, 1,194 Fake)

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Real** | 87.88% | 88.69% | 88.28% | 1,308 |
| **Fake** | 87.48% | 86.60% | 87.04% | 1,194 |

### Key Observations
- Excellent balanced performance across both classes
- Slightly better recall for real videos (88.69% vs 86.60%)
- Strong precision consistency (87.88% vs 87.48%)
- Robust generalization with minimal overfitting

## 7. Future Prospects

### Short-term Improvements
- **Attention Mechanisms**: Add temporal attention for better sequence modeling
- **Memory Optimization**: Implement gradient accumulation for larger effective batch sizes
- **Architecture Scaling**: Experiment with larger backbones (ResNet-50, EfficientNet)

### Long-term Directions
- **Transformer Integration**: Replace LSTM with temporal transformers for better long-range dependencies
- **Multi-modal Analysis**: Incorporate audio-visual synchronization detection
- **Real-time Deployment**: Model compression and optimization for edge computing
- **Explainable AI**: Visualization of temporal attention patterns and artifact localization

## 8. Appendix

### Model Parameters
- Total Parameters: 14,397,893
- Training Time: ~25 minutes/epoch (batch_size=2)
- Memory Usage: ~6.8GB GPU memory
