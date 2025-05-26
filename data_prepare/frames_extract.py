import cv2
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import argparse
import logging
from typing import Dict, List, Tuple
import glob
from tqdm import tqdm
import json

#this class is used to extract frames from videos and generate statistics
class VideoFrameExtractor:
    def __init__(self, input_dir: str, output_dir: str, frames_per_second: float = 1.0):
        """
        Inputs:
            input_dir: directory of input videos
            output_dir: directory to save extracted frames
            frames_per_second: number of frames per second to extract
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.frames_per_second = frames_per_second
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'extraction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.video_stats = []
        
        # video extension in dataset
        self.video_extension = ['.mp4',]
    
    def format_timedelta(self, td: timedelta) -> str:
        """
        format timedelta by replace colons with dashes
        """
        result = str(td)
        try:
            result, ms = result.split(".")
        except ValueError:
            return (result + ".00").replace(":", "-")
        ms = int(ms)
        ms = round(ms / 1e4)
        return f"{result}.{ms:02}".replace(":", "-")
    
    def get_video_info(self, video_path: str) -> Dict:
        """
        extract video properties
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                self.logger.warning(f"error opening: {video_path}")
                return None
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # get size
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            
            # get codec info
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            cap.release()
            
            video_info = {
                'video_path': video_path,
                'filename': os.path.basename(video_path),
                'file_size_mb': round(file_size_mb, 2),
                'duration_seconds': round(duration, 2),
                'fps': round(fps, 2),
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'resolution': f"{width}x{height}",
                'aspect_ratio': round(width/height, 2) if height > 0 else 0,
                'codec': codec.strip('\x00'),
                'bitrate_kbps': round((file_size_mb * 8 * 1024) / duration, 2) if duration > 0 else 0
            }
            
            return video_info
            
        except Exception as e:
            self.logger.error(f"Error getting info for {video_path}: {str(e)}")
            return None
    
    def extract_frames_from_video(self, video_path: str, video_info: Dict) -> Dict:
        """
        Extract frames from a single video at specified rate
        """
        try:
            # Create output folder for this video
            video_name = Path(video_path).stem
            video_output_dir = self.output_dir / video_name
            video_output_dir.mkdir(exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                self.logger.error(f"already opened: {video_path}")
                return {'frames_extracted': 0, 'extraction_success': False}
            
            fps = video_info['fps']
            duration = video_info['duration_seconds']

            if self.frames_per_second <= 0:
                frame_interval = 1  # stride in frames
            else:
                #limit of effective fps
                effective_fps = min(fps, self.frames_per_second)
                frame_interval = max(1, int(fps / effective_fps))
            
            #calculate time step
            time_step = 1.0 / self.frames_per_second if self.frames_per_second > 0 else 1.0 / fps
            
            frames_extracted = 0
            frame_index = 0
            current_time = 0.0
            
            self.logger.info(f"extracting frames from {video_name} (interval: {frame_interval})")
            
            # Extract frames
            while current_time < duration:
                # Set video position to current time
                cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Generate name as timestamp
                td = timedelta(seconds=current_time)
                timestamp = self.format_timedelta(td)
                
                # Save frame
                frame_filename = f"{video_name}_frame_{frames_extracted:06d}_{timestamp}.jpg"
                frame_path = video_output_dir / frame_filename
                
                # Saving frame with JPEG quality
                success = cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                if success:
                    frames_extracted += 1
                else:
                    self.logger.warning(f"failed to save frame: {frame_path}")
                
                current_time += time_step

            cap.release()
            
            extraction_stats = {
                'frames_extracted': frames_extracted,
                'extraction_success': True,
                'output_directory': str(video_output_dir),
                'frame_interval': frame_interval,
                'effective_fps': round(frames_extracted / duration, 2) if duration > 0 else 0
            }
            
            self.logger.info(f"Extracted {frames_extracted} frames from {video_name}")
            return extraction_stats
            
        except Exception as e:
            self.logger.error(f"Error extracting frames from {video_path}: {str(e)}")
            return {'frames_extracted': 0, 'extraction_success': False, 'error': str(e)}
    
    def find_video_files(self) -> List[str]:
        """
        Find all video files in the input directory
        """
        video_files = []
        
        for ext in self.video_extension:
            pattern = str(self.input_dir / f"**/*{ext}")
            video_files.extend(glob.glob(pattern, recursive=True))
            #case of upper extensions
            pattern = str(self.input_dir / f"**/*{ext.upper()}")
            video_files.extend(glob.glob(pattern, recursive=True))
        
        return sorted(list(set(video_files)))  
    
    def process_videos(self) -> None:
        video_files = self.find_video_files()
        
        if not video_files:
            self.logger.warning(f"No files found:{self.input_dir}")
            return
        
        self.logger.info(f"found {len(video_files)} video files")
        
        # progress bar
        for video_path in tqdm(video_files, desc="Processing videos"):
            self.logger.info(f"Processing: {os.path.basename(video_path)}")
            # Get video info
            video_info = self.get_video_info(video_path)
            
            #if error
            if video_info is None:
                self.video_stats.append({
                    'video_path': video_path,
                    'filename': os.path.basename(video_path),
                    'processing_status': 'FAILED - Could not read video',
                    'frames_extracted': 0
                })
                continue
            
            # frames extracted
            extraction_stats = self.extract_frames_from_video(video_path, video_info)
            
            # Combine video info and extraction stats
            combined_stats = {**video_info, **extraction_stats}
            combined_stats['processing_status'] = 'SUCCESS' if extraction_stats['extraction_success'] else 'FAILED'
            combined_stats['frames_per_second_setting'] = self.frames_per_second
            
            self.video_stats.append(combined_stats)
    
    def generate_statistics_report(self) -> None:
        if not self.video_stats:
            self.logger.warning("No video statistics to report")
            return
        
        df = pd.DataFrame(self.video_stats)
        
        # save csv
        csv_path = self.output_dir / "video_extraction_statistics.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Detailed statistics saved to: {csv_path}")

        successful_videos = df[df['processing_status'] == 'SUCCESS']
        
        summary_stats = {
            'extraction_settings': {
                'frames_per_second': self.frames_per_second,
                'input_directory': str(self.input_dir),
                'output_directory': str(self.output_dir)
            },
            'processing_summary': {
                'total_videos_found': len(df),
                'successfully_processed': len(successful_videos),
                'failed_processing': len(df) - len(successful_videos),
                'success_rate': round(len(successful_videos) / len(df) * 100, 2) if len(df) > 0 else 0
            },
            'video_statistics': {},
            'frame_extraction_summary': {},
            'file_size_statistics': {},
            'duration_statistics': {}
        }
        
        if len(successful_videos) > 0:
            summary_stats['video_statistics'] = {
                'total_duration_hours': round(successful_videos['duration_seconds'].sum() / 3600, 2),
                'average_duration_seconds': round(successful_videos['duration_seconds'].mean(), 2),
                'total_file_size_gb': round(successful_videos['file_size_mb'].sum() / 1024, 2),
                'average_file_size_mb': round(successful_videos['file_size_mb'].mean(), 2),
                'resolution_distribution': successful_videos['resolution'].value_counts().to_dict(),
                'fps_statistics': {
                    'mean': round(successful_videos['fps'].mean(), 2),
                    'median': round(successful_videos['fps'].median(), 2),
                    'min': round(successful_videos['fps'].min(), 2),
                    'max': round(successful_videos['fps'].max(), 2)
                }
            }
            
            #frame summary
            summary_stats['frame_extraction_summary'] = {
                'total_frames_extracted': int(successful_videos['frames_extracted'].sum()),
                'average_frames_per_video': round(successful_videos['frames_extracted'].mean(), 2),
                'average_effective_fps': round(successful_videos['effective_fps'].mean(), 2),
                'extraction_efficiency': round(
                    successful_videos['effective_fps'].mean() / self.frames_per_second * 100, 2
                ) if self.frames_per_second > 0 else 100
            }
        
        # Save JSON
        summary_path = self.output_dir / "extraction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        # Print summary to console
        self.print_summary(summary_stats)
        
        self.logger.info(f"Summary statistics saved to: {summary_path}")
    
    def print_summary(self, stats: Dict) -> None:
        """
        Print formatted summary to console
        """
        print("\n" + "="*70)
        print("VIDEO FRAME EXTRACTION SUMMARY")
        print("="*70)
        
        proc_summary = stats['processing_summary']
        print(f"Total Videos: {proc_summary['total_videos_found']}")
        print(f"Successfully Processed: {proc_summary['successfully_processed']}")
        print(f"Failed: {proc_summary['failed_processing']}")
        print(f"Success Rate: {proc_summary['success_rate']}%")
      
        settings = stats['extraction_settings']
        print(f"\nExtraction Settings:")
        print(f"  Frames per Second: {settings['frames_per_second']}")
        print(f"  Input Directory: {settings['input_directory']}")
        print(f"  Output Directory: {settings['output_directory']}")
        
        if 'video_statistics' in stats and stats['video_statistics']:
            vid_stats = stats['video_statistics']
            print(f"\nVideo Statistics:")
            print(f"  Total Duration: {vid_stats['total_duration_hours']} hours")
            print(f"  Average Duration: {vid_stats['average_duration_seconds']} seconds")
            print(f"  Total File Size: {vid_stats['total_file_size_gb']} GB")
            print(f"  Average File Size: {vid_stats['average_file_size_mb']} MB")
            
            fps_stats = vid_stats['fps_statistics']
            print(f"  FPS Range: {fps_stats['min']} - {fps_stats['max']} (avg: {fps_stats['mean']})")
        
        # frame extraction summary 
        if 'frame_extraction_summary' in stats and stats['frame_extraction_summary']:
            frame_stats = stats['frame_extraction_summary']
            print(f"\nFrame Extraction Summary:")
            print(f"  Total Frames Extracted: {frame_stats['total_frames_extracted']:,}")
            print(f"  Average Frames per Video: {frame_stats['average_frames_per_video']}")
            print(f"  Average Effective FPS: {frame_stats['average_effective_fps']}")
            print(f"  Extraction Efficiency: {frame_stats['extraction_efficiency']}%")
        
        print("="*70)

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(
        description="Extract frames from videos and generate statistics"
    )
    parser.add_argument(
        "--input_dir", 
        required=True, 
        help="directory of input videos"
    )
    parser.add_argument(
        "--output_dir", 
        required=True, 
        help="Directory to save extracted report"
    )
    parser.add_argument(
        "--frames_per_second", 
        type=float, 
        default=1.0,
        help="Number of frames to extract per second (default: 1.0)"
    )
    parser.add_argument(
        "--recursive", 
        action="store_true",
        help="Search for videos recursively in subdirectories"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    # Initialize extractor
    extractor = VideoFrameExtractor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        frames_per_second=args.frames_per_second
    )
    
    # Process videos
    print(f"Starting frame extraction from: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Frames per second: {args.frames_per_second}")
    
    extractor.process_videos()
    extractor.generate_statistics_report()
    
    print(f"\nExtraction complete! Check '{args.output_dir}' for results.")

if __name__ == "__main__":
    main()
"""example usage
python frames_extract.py --input_dir "/path/to/videos" --output_dir "/path/to/output" --frames_per_second 1.0
"""