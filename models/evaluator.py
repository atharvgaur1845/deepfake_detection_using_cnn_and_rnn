import torch
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class DeepfakeEvaluator:
    def __init__(self, model, test_loader, device, output_dir="evaluation_results"):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self):
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for sequences, labels in self.test_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        #evaluation metrics
        self.generate_classification_report(all_labels, all_predictions)
        self.plot_confusion_matrix(all_labels, all_predictions)
        self.plot_roc_curve(all_labels, all_probabilities)
        self.plot_precision_recall_curve(all_labels, all_probabilities)
        
        # Calculate and save detailed metrics
        metrics = self.calculate_detailed_metrics(all_labels, all_predictions, all_probabilities)
        
        with open(self.output_dir / "evaluation_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Evaluation completed. Results saved to {self.output_dir}")
        return metrics
    
    def calculate_detailed_metrics(self, labels, predictions, probabilities):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        f1 = f1_score(labels, predictions, average='weighted')
        
        #calculate AUC
        if len(np.unique(labels)) == 2:
            auc = roc_auc_score(labels, probabilities[:, 1])
        else:
            auc = 0.0
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'total_samples': len(labels),
            'class_distribution': {
                'real': int(np.sum(labels == 0)),
                'fake': int(np.sum(labels == 1))
            }
        }
        
        return metrics
    
    def generate_classification_report(self, labels, predictions):
        """Generate and save classification report"""
        report = classification_report(labels, predictions, 
                                     target_names=['Real', 'Fake'],
                                     output_dict=True)
        
        # Save text report
        report_text = classification_report(labels, predictions, 
                                          target_names=['Real', 'Fake'])
        
        with open(self.output_dir / "classification_report.txt", 'w') as f:
            f.write(report_text)
    
        with open(self.output_dir / "classification_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print("Classification Report:")
        print(report_text)
    
    def plot_confusion_matrix(self, labels, predictions):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], 
                   yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, labels, probabilities):
        if len(np.unique(labels)) == 2:
            fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
            auc = roc_auc_score(labels, probabilities[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.output_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_precision_recall_curve(self, labels, probabilities):
        if len(np.unique(labels)) == 2:
            precision, recall, _ = precision_recall_curve(labels, probabilities[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.output_dir / "precision_recall_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
