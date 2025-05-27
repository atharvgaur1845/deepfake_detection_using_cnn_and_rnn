import os
from pathlib import Path
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DeepfakeNumpyDataset(Dataset):
    def __init__(self, data_root_dir: str, sequence_length: int = 16, 
                 label_mapping: Dict[str, int] = None, split: str = 'train'):
        """
        Initialize dataset
        Args:
            data_root_dir: Root directory containing processed .npy files
            sequence_length: Number of frames per sequence
            label_mapping: Dict mapping folder names to labels {'real': 0, 'fake': 1}
            split: 'train' or 'test'
        """
        self.data_root_dir = Path(data_root_dir)
        self.sequence_length = sequence_length
        self.label_mapping = label_mapping or {'real': 0, 'fake': 1}
        self.split = split
    
        self.sequences = self._load_sequences()
        
        print(f"Loaded {len(self.sequences)} sequences for {split}")
    
    def _load_sequences(self) -> List[Dict]:
        """Load all available sequences from .npy files"""
        sequences = []

        for class_name in self.label_mapping.keys():
            class_dir = self.data_root_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue
            
            label = self.label_mapping[class_name]
        
            video_dirs = [d for d in class_dir.iterdir() if d.is_dir()]
            
            for video_dir in video_dirs:
                video_name = video_dir.name
            
                face_files = []
                
                #find all face subdirectories
                face_dirs = [d for d in video_dir.iterdir() if d.is_dir()]
                
                for face_dir in face_dirs:
                    # get all .npy files
                    npy_files = sorted(face_dir.glob("*.npy"))
                    face_files.extend(npy_files)
                
                # Sort files
                face_files = sorted(face_files)
                
                if len(face_files) >= self.sequence_length:
                    #create overlapping sequences
                    step_size = max(1, self.sequence_length // 4)  # 75% overlap
                    
                    for i in range(0, len(face_files) - self.sequence_length + 1, step_size):
                        sequence_files = face_files[i:i + self.sequence_length]
                        
                        sequences.append({
                            'files': sequence_files,
                            'label': label,
                            'video_name': video_name,
                            'class_name': class_name
                        })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_info = self.sequences[idx]
    
        sequence_data = []
        
        for npy_file in sequence_info['files']:
            try:
                # Load normalized face data
                face_data = np.load(npy_file)
                
                # Ensure correct shape (H, W, C)
                if len(face_data.shape) == 2:  # Grayscale
                    face_data = np.expand_dims(face_data, axis=-1)
                elif len(face_data.shape) == 3 and face_data.shape[0] == 3:  # (C, H, W)
                    face_data = np.transpose(face_data, (1, 2, 0))  # Convert to (H, W, C)
                
                sequence_data.append(face_data)
                
            except Exception as e:
                print(f"Error loading {npy_file}: {e}")
                # Create dummy data if file is corrupted
                dummy_data = np.zeros((224, 224, 3), dtype=np.float32)
                sequence_data.append(dummy_data)
        
        # Convert to tensor: (sequence_length, height, width, channels)
        sequence_tensor = torch.tensor(np.array(sequence_data), dtype=torch.float32)
        
        # Rearrange to (sequence_length, channels, height, width)
        sequence_tensor = sequence_tensor.permute(0, 3, 1, 2)
        
        label_tensor = torch.tensor(sequence_info['label'], dtype=torch.long)
        
        return sequence_tensor, label_tensor

def create_data_loaders(train_data_dir: str, test_data_dir: str, batch_size: int = 8, 
                       sequence_length: int = 16, num_workers: int = 4):
    """Args:
        train_data_dir:training processed data
        test_data_dir:test processed data
        batch_size: Batch size for training
        sequence_length: Number of frames per sequence
        num_workers: Number of worker processes for data loading
    
    """
    
    # Define label mapping
    label_mapping = {'real': 0, 'fake': 1}
    
    # Create datasets
    train_dataset = DeepfakeNumpyDataset(
        data_root_dir=train_data_dir,
        sequence_length=sequence_length,
        label_mapping=label_mapping,
        split='train'
    )
    
    test_dataset = DeepfakeNumpyDataset(
        data_root_dir=test_data_dir,
        sequence_length=sequence_length,
        label_mapping=label_mapping,
        split='test'
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, test_loader
