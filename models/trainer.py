import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class DeepfakeTrainer:
    def __init__(self, model, train_loader, test_loader, device, 
                 learning_rate=0.001, output_dir="results"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        #oss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        #training history
        self.history = {
            'train_losses': [],
            'test_losses': [],
            'test_accuracies': [],
            'learning_rates': []
        }
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        for sequences, labels in tqdm(self.train_loader, desc="Training"):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            running_loss += loss.item()
        
        return running_loss / len(self.train_loader)
    
    def test(self):
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in tqdm(self.test_loader, desc="Testing"):
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                # Get predictions
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        test_loss = running_loss / len(self.test_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return test_loss, accuracy, precision, recall, f1
    
    def train(self, num_epochs: int):
        """Train the model for specified epochs"""
        best_accuracy = 0.0
        best_model_path = self.output_dir / "best_model.pth"
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch()
            
            # Test
            test_loss, accuracy, precision, recall, f1 = self.test()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save metrics
            self.history['train_losses'].append(train_loss)
            self.history['test_losses'].append(test_loss)
            self.history['test_accuracies'].append(accuracy)
            self.history['learning_rates'].append(current_lr)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")
            print(f"Test F1-Score: {f1:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'accuracy': accuracy,
                    'loss': test_loss
                }, best_model_path)
                print(f"New best model saved! Accuracy: {best_accuracy:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'accuracy': accuracy,
                    'loss': test_loss
                }, checkpoint_path)
        
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        metadata = {
            'best_accuracy': best_accuracy,
            'total_epochs': num_epochs,
            'final_lr': current_lr,
            'model_architecture': 'HybridDeepFakeDetector'
        }
        
        with open(self.output_dir / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.plot_training_curves()
        
        print(f"\nTraining completed! Best accuracy: {best_accuracy:.4f}")
        return best_accuracy
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_losses']) + 1)

        ax1.plot(epochs, self.history['train_losses'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.history['test_losses'], 'r-', label='Test Loss')
        ax1.set_title('Training and Test Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        ax2.plot(epochs, self.history['test_accuracies'], 'g-', label='Test Accuracy')
        ax2.set_title('Test Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate
        ax3.plot(epochs, self.history['learning_rates'], 'm-', label='Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
        
        # Overfitting monitor
        loss_diff = [test - train for test, train in zip(self.history['test_losses'], self.history['train_losses'])]
        ax4.plot(epochs, loss_diff, 'orange', label='Test - Train Loss')
        ax4.set_title('Overfitting Monitor')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Loss Difference')
        ax4.legend()
        ax4.grid(True)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
