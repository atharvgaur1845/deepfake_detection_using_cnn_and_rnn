import cv2
import dlib
import numpy as np
import os
from pathlib import Path
import argparse
from imutils import face_utils
import json
from tqdm import tqdm

class SimpleFaceDetector:
    def __init__(self, predictor_path: str, output_size: tuple = (224, 224)):
        """    
        Args:
            predictor_path: Path to dlib shape predictor
            output_size: Size of extracted faces (width, height)
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.output_size = output_size
        
    def detect_and_extract_face(self, image):
        """
        Args:
            image: Input image (BGR format)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None
        
        # getting largest face
        largest_face = max(faces, key=lambda rect: (rect.right() - rect.left()) * (rect.bottom() - rect.top()))
        
        #face landmarks for alignment
        landmarks = self.predictor(gray, largest_face)
        landmarks = face_utils.shape_to_np(landmarks)
        
        # Align face using eye landmarks
        aligned_face = self.align_face(image, landmarks)
        
        if aligned_face is not None:
            return aligned_face
        else:
            # Fallback to simple crop
            return self.simple_crop_face(image, largest_face)
    
    def align_face(self, image, landmarks):
        """
        Align face using eye landmarks
        """
        try:
            #get eye points
            left_eye = landmarks[36:42] 
            right_eye = landmarks[42:48] 
            left_eye_center = left_eye.mean(axis=0).astype("int")
            right_eye_center = right_eye.mean(axis=0).astype("int")
            
            # calculate angle between eyes
            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            #calculate center point between eyes
            eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                          (left_eye_center[1] + right_eye_center[1]) // 2)
            M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
            
            # Apply rotation
            aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            
            # Crop and resize face region
            face_width = int(abs(right_eye_center[0] - left_eye_center[0]) * 3)
            face_height = int(face_width * 1.2)
            
            x = max(0, eyes_center[0] - face_width // 2)
            y = max(0, eyes_center[1] - face_height // 3)
            
            face_crop = aligned[y:y+face_height, x:x+face_width]
            
            if face_crop.size > 0:
                face_resized = cv2.resize(face_crop, self.output_size)
                return face_resized
            else:
                return None
                
        except:
            return None
    
    def simple_crop_face(self, image, face_rect):
        """
        Simple face cropping without alignment
        """
        x, y, w, h = face_utils.rect_to_bb(face_rect)
        
        # Add padding
        padding = 0.3
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image.shape[1], x + w + pad_x)
        y2 = min(image.shape[0], y + h + pad_y)
        
        face_crop = image[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, self.output_size)
        
        return face_resized
    
    def process_video_frames(self, frames_dir, output_dir, video_name):
        """
        Args:
            frames_dir: Directory containing video frames
            output_dir: Output directory for extracted faces
            video_name: Name of the video (for organizing output)
        """
        frames_path = Path(frames_dir)
        output_path = Path(output_dir)
        
        # Create video-specific output directory
        video_output_dir = output_path / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files in the video folder
        frame_files = []
        for ext in ['.jpg', ]:
            frame_files.extend(frames_path.glob(f"*{ext}"))
            frame_files.extend(frames_path.glob(f"*{ext.upper()}"))
        
        frame_files = sorted(frame_files)
        
        if not frame_files:
            print(f"No frames found in folder: {frames_dir}")
            return []
        
        extracted_faces = []
        face_count = 0
        
        print(f"Processing {len(frame_files)} frames for video: {video_name}")
        
        for frame_file in tqdm(frame_files, desc=f"Extracting faces from {video_name}"):
            # load frame
            image = cv2.imread(str(frame_file))
            if image is None:
                continue
            
            # extract face
            face = self.detect_and_extract_face(image)
            
            if face is not None:
                # save extracted face
                face_filename = f"face_{face_count:06d}.jpg"
                face_path = video_output_dir / face_filename
                cv2.imwrite(str(face_path), face)
                
                extracted_faces.append({
                    'original_frame': str(frame_file),
                    'face_path': str(face_path),
                    'face_index': face_count
                })
                face_count += 1
        
        print(f"Extracted {face_count} faces from {video_name}")
        
        # Save extraction log
        log_file = video_output_dir / "extraction_log.json"
        with open(log_file, 'w') as f:
            json.dump({
                'video_name': video_name,
                'total_frames': len(frame_files),
                'extracted_faces': face_count,
                'extraction_rate': face_count / len(frame_files) if frame_files else 0,
                'faces': extracted_faces
            }, f, indent=2)
        
        return extracted_faces
    
    def process_all_videos(self, frames_root_dir, output_root_dir):
        """
        Args:
            frames_root_dir: Root directory containing video frame folders
            output_root_dir: Root output directory for all extracted faces
        """
        frames_root = Path(frames_root_dir)
        output_root = Path(output_root_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        
        # find all subdirectories
        video_folders = [d for d in frames_root.iterdir() if d.is_dir()]
        
        if not video_folders:
            print(f"No video folders found in {frames_root_dir}")
            return {}
        
        print(f"Found {len(video_folders)} video folders to process")
        
        all_results = {}
        total_faces_extracted = 0
        
        for video_folder in video_folders:
            video_name = video_folder.name
            print(f"\n{'='*60}")
            print(f"Processing video folder: {video_name}")
            print(f"{'='*60}")
            
            faces = self.process_video_frames(
                frames_dir=str(video_folder),
                output_dir=str(output_root),
                video_name=video_name
            )
            
            all_results[video_name] = {
                'extracted_faces': len(faces),
                'face_details': faces
            }
            total_faces_extracted += len(faces)
        
        # Save overall summary
        summary = {
            'total_videos_processed': len(video_folders),
            'total_faces_extracted': total_faces_extracted,
            'average_faces_per_video': total_faces_extracted / len(video_folders) if video_folders else 0,
            'video_results': all_results
        }
        
        summary_file = output_root / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total videos processed: {len(video_folders)}")
        print(f"Total faces extracted: {total_faces_extracted}")
        print(f"Average faces per video: {total_faces_extracted / len(video_folders):.2f}")
        print(f"Results saved to: {output_root}")
        print(f"Summary saved to: {summary_file}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Extract faces from all video frame folders")
    parser.add_argument("--frames_root_dir", required=True, 
                       help="Root directory containing video frame folders")
    parser.add_argument("--output_root_dir", required=True, 
                       help="Root output directory for extracted faces")
    parser.add_argument("--predictor_path", required=True, 
                       help="Path to dlib shape predictor")
    parser.add_argument("--output_size", type=int, nargs=2, default=[224, 224], 
                       help="Face size (width height)")
    
    args = parser.parse_args()

    if not os.path.exists(args.frames_root_dir):
        print(f"Error: Frames root directory does not exist: {args.frames_root_dir}")
        return

    if not os.path.exists(args.predictor_path):
        print(f"Error: Predictor file does not exist: {args.predictor_path}")
        return

    # Initialize detector
    detector = SimpleFaceDetector(
        predictor_path=args.predictor_path,
        output_size=tuple(args.output_size)
    )

    print(f"Input directory: {args.frames_root_dir}")
    print(f"Output directory: {args.output_root_dir}")
    print(f"Face size: {args.output_size[0]}x{args.output_size[1]}")
    
    detector.process_all_videos(
        frames_root_dir=args.frames_root_dir,
        output_root_dir=args.output_root_dir
    )

if __name__ == "__main__":
    main()
# command to run:
#python face_detection.py \
    #--frames_root_dir "/path/to/all_video_frames/" \
    #--output_root_dir "/path/to/all_extracted_faces/" \
    #--predictor_path "shape_predictor_68_face_landmarks.dat"