import torch
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
import sys
from tqdm import tqdm

class DeepfakeEvaluator:
    def __init__(self, model_path, test_loader, device, output_dir="evaluation_results"):
        """
        Initialize evaluator with model path instead of model object
        
        Args:
            model_path: Path to saved model (.pth file)
            test_loader: DataLoader for test data
            device: torch device
            output_dir: Directory to save evaluation results
        """
        self.device = device
        self.test_loader = test_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model from path
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load model from .pth file"""
        # Import model here to avoid circular imports
        from .hybrid_model import HybridDeepFakeDetector
        
        print(f"Loading model from: {model_path}")
        
        # Create model instance
        model = HybridDeepFakeDetector(
            input_channels=3,
            cnn_feature_dim=512,
            lstm_hidden_dim=256,
            num_classes=2
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Training accuracy: {checkpoint.get('accuracy', 'unknown'):.4f}")
            print(f"Training loss: {checkpoint.get('loss', 'unknown'):.4f}")
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
            print("Model state dict loaded directly")
        
        model.to(self.device)
        model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def evaluate(self):
        """Run comprehensive evaluation"""
        print("Starting model evaluation...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for sequences, labels in tqdm(self.test_loader, desc="Evaluating"):
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Generate evaluation metrics
        self.generate_classification_report(all_labels, all_predictions)
        self.plot_confusion_matrix(all_labels, all_predictions)
        self.plot_roc_curve(all_labels, all_probabilities)
        self.plot_precision_recall_curve(all_labels, all_probabilities)
        
        # Calculate and save detailed metrics
        metrics = self.calculate_detailed_metrics(all_labels, all_predictions, all_probabilities)
        
        # Save detailed predictions for analysis
        self.save_detailed_results(all_labels, all_predictions, all_probabilities)
        
        with open(self.output_dir / "evaluation_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nEvaluation completed. Results saved to {self.output_dir}")
        return metrics
    
    def calculate_detailed_metrics(self, labels, predictions, probabilities):
        """Calculate comprehensive evaluation metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
        
        # Calculate AUC for binary classification
        if len(np.unique(labels)) == 2:
            auc = roc_auc_score(labels, probabilities[:, 1])
        else:
            auc = 0.0
        
        # Calculate per-class accuracy
        class_accuracies = {}
        for class_idx in [0, 1]:
            class_mask = labels == class_idx
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(labels[class_mask], predictions[class_mask])
                class_accuracies[f'class_{class_idx}'] = float(class_acc)
        
        metrics = {
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc': float(auc)
            },
            'per_class_metrics': {
                'real': {
                    'precision': float(precision_per_class[0]),
                    'recall': float(recall_per_class[0]),
                    'f1_score': float(f1_per_class[0]),
                    'accuracy': class_accuracies.get('class_0', 0.0)
                },
                'fake': {
                    'precision': float(precision_per_class[1]),
                    'recall': float(recall_per_class[1]),
                    'f1_score': float(f1_per_class[1]),
                    'accuracy': class_accuracies.get('class_1', 0.0)
                }
            },
            'dataset_info': {
                'total_samples': len(labels),
                'class_distribution': {
                    'real': int(np.sum(labels == 0)),
                    'fake': int(np.sum(labels == 1))
                },
                'class_balance': float(np.sum(labels == 0) / len(labels))
            }
        }
        
        return metrics
    
    def save_detailed_results(self, labels, predictions, probabilities):
        """Save detailed prediction results for further analysis"""
        results = {
            'true_labels': labels.tolist(),
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'correct_predictions': (labels == predictions).tolist()
        }
        
        with open(self.output_dir / "detailed_predictions.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    def generate_classification_report(self, labels, predictions):
        """Generate and save classification report"""
        report = classification_report(labels, predictions, 
                                     target_names=['Real', 'Fake'],
                                     output_dict=True, zero_division=0)
        
        # Save text report
        report_text = classification_report(labels, predictions, 
                                          target_names=['Real', 'Fake'], zero_division=0)
        
        with open(self.output_dir / "classification_report.txt", 'w') as f:
            f.write(report_text)
        
        with open(self.output_dir / "classification_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\nClassification Report:")
        print(report_text)
    
    def plot_confusion_matrix(self, labels, predictions):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], 
                   yticklabels=['Real', 'Fake'], cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        
        # Add percentage annotations
        total = cm.sum()
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                percentage = cm[i][j] / total * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, labels, probabilities):
        """Plot and save ROC curve"""
        if len(np.unique(labels)) == 2:
            fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
            auc = roc_auc_score(labels, probabilities[:, 1])
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=3, 
                    label=f'ROC curve (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=14)
            plt.ylabel('True Positive Rate', fontsize=14)
            plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
            plt.legend(loc="lower right", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_precision_recall_curve(self, labels, probabilities):
        """Plot and save Precision-Recall curve"""
        if len(np.unique(labels)) == 2:
            precision, recall, _ = precision_recall_curve(labels, probabilities[:, 1])
            
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, color='blue', lw=3, label='Precision-Recall Curve')
            plt.xlabel('Recall', fontsize=14)
            plt.ylabel('Precision', fontsize=14)
            plt.title('Precision-Recall Curve', fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.tight_layout()
            plt.savefig(self.output_dir / "precision_recall_curve.png", dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Evaluate DeepFake Detection Model")
    parser.add_argument("--model_path", required=True, help="Path to saved model (.pth file)")
    parser.add_argument("--test_data_dir", required=True, help="Test data directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for evaluation results")
    parser.add_argument("--sequence_length", type=int, default=16, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    args = parser.parse_args()
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Import data loader
    from dataset_loader import create_data_loaders
    
    # Create test data loader
    print("Creating test data loader...")
    _, test_loader = create_data_loaders(
        train_data_dir=args.test_data_dir,  # Dummy, not used
        test_data_dir=args.test_data_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_workers=args.num_workers
    )
    
    # Create evaluator and run evaluation
    evaluator = DeepfakeEvaluator(
        model_path=args.model_path,
        test_loader=test_loader,
        device=device,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    overall = metrics['overall_metrics']
    dataset_info = metrics['dataset_info']
    
    print(f"Overall Accuracy: {overall['accuracy']:.4f}")
    print(f"Overall Precision: {overall['precision']:.4f}")
    print(f"Overall Recall: {overall['recall']:.4f}")
    print(f"Overall F1-Score: {overall['f1_score']:.4f}")
    print(f"AUC-ROC: {overall['auc']:.4f}")
    
    print(f"\nDataset Information:")
    print(f"Total Samples: {dataset_info['total_samples']}")
    print(f"Real Samples: {dataset_info['class_distribution']['real']}")
    print(f"Fake Samples: {dataset_info['class_distribution']['fake']}")
    print(f"Class Balance: {dataset_info['class_balance']:.2%} real")
    
    print(f"\nPer-Class Performance:")
    real_metrics = metrics['per_class_metrics']['real']
    fake_metrics = metrics['per_class_metrics']['fake']
    
    print(f"Real Class - Precision: {real_metrics['precision']:.4f}, "
          f"Recall: {real_metrics['recall']:.4f}, F1: {real_metrics['f1_score']:.4f}")
    print(f"Fake Class - Precision: {fake_metrics['precision']:.4f}, "
          f"Recall: {fake_metrics['recall']:.4f}, F1: {fake_metrics['f1_score']:.4f}")
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
