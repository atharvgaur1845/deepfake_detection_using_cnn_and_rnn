import torch
from pathlib import Path
import argparse
import sys
sys.path.append(str(Path(__file__).parent))

from dataset_loader import create_data_loaders
from hybrid_model import HybridDeepFakeDetector
from trainer import DeepfakeTrainer

def main():
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument("--train_data_dir", required=True, help="Training data directory")
    parser.add_argument("--test_data_dir", required=True, help="Test data directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--sequence_length", type=int, default=16, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, test_loader = create_data_loaders(
        train_data_dir=args.train_data_dir,
        test_data_dir=args.test_data_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_workers=args.num_workers
    )
    
    # Create model
    model = HybridDeepFakeDetector(
        input_channels=3,
        cnn_feature_dim=512,
        lstm_hidden_dim=256,
        num_classes=2
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = DeepfakeTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
    
    # Train model
    best_accuracy = trainer.train(args.num_epochs)
    
    print(f"Training completed. Best accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()
