import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import json
import random
from tqdm import tqdm
import albumentations as A

class DataAugmentationNormalization:
    
    def __init__(self, normalization_range="0_1", augmentation_factor=3):
        """
        Args:
            normalization_range: "0_1" for [0,1] or "-1_1" for [-1,1]
            augmentation_factor: Number of augmented versions per image
        """
        self.normalization_range = normalization_range
        self.augmentation_factor = augmentation_factor
        
        # Setup augmentation pipeline
        self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
        ])
    
    def normalize_image(self, image):
        """
        Args:
            image: Input image (0-255)
            
        Returns:
            Normalized image
        """
        image_float = image.astype(np.float32)
        
        if self.normalization_range == "0_1":
            return image_float / 255.0
        elif self.normalization_range == "-1_1":
            return (image_float / 255.0) * 2.0 - 1.0
        else:
            raise ValueError("normalization_range must be '0_1' or '-1_1'")
    
    def denormalize_for_save(self, normalized_image):
        """
        Convert normalized image back to 0-255 for saving
        """
        if self.normalization_range == "0_1":
            return (normalized_image * 255.0).astype(np.uint8)
        elif self.normalization_range == "-1_1":
            return ((normalized_image + 1.0) / 2.0 * 255.0).astype(np.uint8)
    
    def apply_augmentation(self, image):
        """
        Args:
            image: Input image (0-255)
        """
        try:
            augmented = self.augmentation_pipeline(image=image)
            return augmented["image"]
        except:
            return image
    
    def create_simple_augmentations(self, image):
        """
        Args:
            image: Input image
        """
        augmented_images = []
        
        # original image
        augmented_images.append(("original", image.copy()))
        
        # flipping image
        flipped = cv2.flip(image, 1)
        augmented_images.append(("flipped", flipped))
        
        # making variations
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
        dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-20)
        augmented_images.append(("bright", bright))
        augmented_images.append(("dark", dark))
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        augmented_images.append(("blurred", blurred))
        
        return augmented_images
    
    def process_face_directory(self, input_dir, output_dir, apply_augmentation=True):
        """
        Args:
            input_dir: Directory containing face images
            output_dir: Output directory for processed images
            apply_augmentation: Whether to apply augmentation
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all face images
        face_files = []
        for ext in ['.jpg',]:
            face_files.extend(input_path.glob(f"*{ext}"))
            face_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        face_files = sorted(face_files)
        
        if not face_files:
            print(f"No face images found in {input_dir}")
            return 0
        
        print(f"Processing {len(face_files)} face images...")
        
        processed_count = 0
        
        for face_file in tqdm(face_files, desc="Processing faces"):
            image = cv2.imread(str(face_file))
            if image is None:
                continue
            
            # Create subdirectory for this face
            face_name = face_file.stem
            face_output_dir = output_path / face_name
            face_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Normalize original image
            normalized_image = self.normalize_image(image)
            
            # Save normalized original
            original_path = face_output_dir / "normalized_original.npy"
            np.save(original_path, normalized_image)
            
            if apply_augmentation:
                # Apply albumentations augmentation
                for i in range(self.augmentation_factor):
                    augmented = self.apply_augmentation(image)
                    normalized_aug = self.normalize_image(augmented)
                    
                    aug_path = face_output_dir / f"normalized_aug_{i:02d}.npy"
                    np.save(aug_path, normalized_aug)
                
                # Apply simple augmentations
                simple_augs = self.create_simple_augmentations(image)
                for aug_name, aug_image in simple_augs[1:]:  # Skip original
                    normalized_aug = self.normalize_image(aug_image)
                    
                    aug_path = face_output_dir / f"normalized_{aug_name}.npy"
                    np.save(aug_path, normalized_aug)
            
            processed_count += 1
        
        print(f"Processed {processed_count} face images")
        
        # Save processing info
        info = {
            'total_faces_processed': processed_count,
            'normalization_range': self.normalization_range,
            'augmentation_applied': apply_augmentation,
            'augmentation_factor': self.augmentation_factor if apply_augmentation else 0
        }
        
        with open(output_path / "processing_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        return processed_count
    
    def process_all_video_face_folders(self, root_input_dir, root_output_dir, apply_augmentation=True):
        """
        Args:
            root_input_dir: Root directory containing video face folders
            root_output_dir: Root output directory for processed data
            apply_augmentation: Whether to apply augmentation
        """
        root_input_path = Path(root_input_dir)
        root_output_path = Path(root_output_dir)
        root_output_path.mkdir(parents=True, exist_ok=True)

        video_folders = [d for d in root_input_path.iterdir() if d.is_dir()]
        
        if not video_folders:
            print(f"No video folders found in {root_input_dir}")
            return {}
        
        print(f"Found {len(video_folders)} video folders to process")
        
        all_results = {}
        total_faces_processed = 0
        
        for video_folder in video_folders:
            video_name = video_folder.name
            print(f"\n{'='*60}")
            print(f"Processing video folder: {video_name}")
            print(f"{'='*60}")
            
            video_output_dir = root_output_path / video_name
            
            # Processing faces
            faces_processed = self.process_face_directory(
                input_dir=str(video_folder),
                output_dir=str(video_output_dir),
                apply_augmentation=apply_augmentation
            )
            
            all_results[video_name] = {
                'faces_processed': faces_processed,
                'input_path': str(video_folder),
                'output_path': str(video_output_dir)
            }
            total_faces_processed += faces_processed
        
        # Save overall summary
        summary = {
            'total_videos_processed': len(video_folders),
            'total_faces_processed': total_faces_processed,
            'average_faces_per_video': total_faces_processed / len(video_folders) if video_folders else 0,
            'normalization_range': self.normalization_range,
            'augmentation_applied': apply_augmentation,
            'augmentation_factor': self.augmentation_factor if apply_augmentation else 0,
            'video_results': all_results
        }
        
        summary_file = root_output_path / "overall_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total videos processed: {len(video_folders)}")
        print(f"Total faces processed: {total_faces_processed}")
        print(f"Average faces per video: {total_faces_processed / len(video_folders):.2f}")
        print(f"Normalization range: {self.normalization_range}")
        print(f"Augmentation applied: {apply_augmentation}")
        if apply_augmentation:
            print(f"Augmentation factor: {self.augmentation_factor}")
        print(f"Results saved to: {root_output_path}")
        print(f"Summary saved to: {summary_file}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Augment and normalize all video face folders")
    parser.add_argument("--root_input_dir", required=True, 
                       help="Root directory containing video face folders")
    parser.add_argument("--root_output_dir", required=True, 
                       help="Root output directory for processed data")
    parser.add_argument("--normalization", choices=["0_1", "-1_1"], default="0_1",
                       help="Normalization range")
    parser.add_argument("--augmentation_factor", type=int, default=3, 
                       help="Number of augmentations per image")
    parser.add_argument("--no_augmentation", action="store_true", 
                       help="Skip augmentation")
    
    args = parser.parse_args()
    
    # check input
    if not os.path.exists(args.root_input_dir):
        print(f"Error: Input directory does not exist: {args.root_input_dir}")
        return
    
    processor = DataAugmentationNormalization(
        normalization_range=args.normalization,
        augmentation_factor=args.augmentation_factor
    )
    print(f"Starting augmentation and normalization for all videos...")
    print(f"Input directory: {args.root_input_dir}")
    print(f"Output directory: {args.root_output_dir}")
    print(f"Normalization: {args.normalization}")
    print(f"Augmentation: {'Disabled' if args.no_augmentation else 'Enabled'}")
    if not args.no_augmentation:
        print(f"Augmentation factor: {args.augmentation_factor}")
    
    processor.process_all_video_face_folders(
        root_input_dir=args.root_input_dir,
        root_output_dir=args.root_output_dir,
        apply_augmentation=not args.no_augmentation
    )

if __name__ == "__main__":
    main()
#command to run the script:
#python augmentation_normalization.py \
    #--root_input_dir "/path/to/extracted_faces/train/real" \
    #--root_output_dir "/path/to/processed_data/train/real"