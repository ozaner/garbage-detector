"""
Garbage truck safety detection system.

This application analyzes video footage from a garbage truck's point of view
to detect potential safety issues when collecting trash.
"""
import argparse
import json
import sys
import time
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import cv2

from src.detection.safety_analyzer import SafetyAnalyzer
from src.video.processor import VideoProcessor


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze garbage truck footage for safety issues"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the video file to analyze"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="safety_report.json",
        help="Path to save the safety report (default: safety_report.json)"
    )
    parser.add_argument(
        "-n", "--frame-interval",
        type=int,
        default=30,
        help="Analyze every n-th frame (default: 30)"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save frames with detected safety issues"
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="detected_frames",
        help="Directory to save frames with detected issues (default: detected_frames)"
    )
    parser.add_argument(
        "--save-all-frames",
        action="store_true",
        help="Save all analyzed frames regardless of safety issues"
    )
    parser.add_argument(
        "--all-frames-dir",
        type=str,
        default="all_frames",
        help="Directory to save all analyzed frames (default: all_frames)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for frame analysis (default: 4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of frames to process in each batch (default: 10)"
    )
    
    return parser.parse_args()


def prompt_for_frame_interval():
    """Prompt the user for the frame interval."""
    while True:
        try:
            interval = int(input("Enter frame interval (analyze every n-th frame, e.g. 30): "))
            if interval <= 0:
                print("Interval must be a positive integer.")
                continue
            return interval
        except ValueError:
            print("Please enter a valid number.")


def analyze_frame_task(args):
    """
    Task function for parallel processing of frames.
    
    Args:
        args (tuple): Tuple containing (frame_number, frame, timestamp, analyzer, all_frames_dir)
        
    Returns:
        tuple: (frame_number, timestamp, analysis_result, frame)
    """
    frame_number, frame, timestamp, analyzer, all_frames_dir = args
    
    # Save the frame if save_all_frames is enabled
    if all_frames_dir:
        frame_filename = f"frame_{frame_number:06d}_{timestamp.replace(':', '_')}.jpg"
        frame_path = all_frames_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)
    
    # Analyze the frame for safety issues
    analysis_result = analyzer.analyze_frame(frame)
    
    return frame_number, timestamp, analysis_result, frame


def main():
    """Run the garbage truck safety detection system."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if the video file exists
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Only prompt for frame interval if it wasn't specified in command line arguments
    if args.frame_interval is None:
        frame_interval = prompt_for_frame_interval()
    else:
        frame_interval = args.frame_interval
    
    print(f"Analyzing every {frame_interval}th frame...")
    
    # Initialize the video processor
    try:
        video_processor = VideoProcessor(video_path)
    except Exception as e:
        print(f"Error initializing video processor: {e}")
        sys.exit(1)
    
    # Print video information
    print("\nVideo Information:")
    print(f"Duration: {video_processor.get_frame_timestamp(video_processor.frame_count - 1)}")
    print(f"Frame count: {video_processor.frame_count}")
    print(f"FPS: {video_processor.fps}")
    print(f"Resolution: {video_processor.width}x{video_processor.height}")
    
    # Calculate number of frames to process
    num_frames_to_process = video_processor.frame_count // frame_interval
    if video_processor.frame_count % frame_interval > 0:
        num_frames_to_process += 1
    
    print(f"Analyzing {num_frames_to_process} frames with {args.workers} parallel workers...\n")
    
    # Initialize the safety analyzer
    try:
        safety_analyzer = SafetyAnalyzer()
    except Exception as e:
        print(f"Error initializing safety analyzer: {e}")
        sys.exit(1)
    
    # Create output directory for frames if needed
    frames_dir = None
    if args.save_frames:
        frames_dir = Path(args.frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"Frames with detected issues will be saved to: {frames_dir}")
    
    # Create output directory for all frames if needed
    all_frames_dir = None
    if args.save_all_frames:
        all_frames_dir = Path(args.all_frames_dir)
        all_frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"All analyzed frames will be saved to: {all_frames_dir}")
    
    # Process the video and analyze frames
    safety_report = {
        "video_file": str(video_path),
        "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame_interval": frame_interval,
        "detected_issues": []
    }
    
    # Prepare frame batches for parallel processing
    frame_batches = []
    for frame_number, frame in video_processor.get_frame_at_intervals(frame_interval):
        timestamp = video_processor.get_frame_timestamp(frame_number)
        frame_batches.append((frame_number, frame, timestamp, safety_analyzer, all_frames_dir))
    
    # Process frames in parallel
    issues_count = 0
    
    # Use ThreadPoolExecutor for I/O-bound operations (API calls)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks and get futures
        with tqdm(total=len(frame_batches), desc="Analyzing frames") as progress_bar:
            # Process results as they complete
            for frame_number, timestamp, analysis_result, frame in executor.map(analyze_frame_task, frame_batches):
                # Process the results
                safety_issues = analysis_result.get("safety_issues", [])
                if safety_issues:
                    issues_count += len(safety_issues)
                    
                    # Add the issues to the report
                    for issue in safety_issues:
                        safety_report["detected_issues"].append({
                            "frame_number": frame_number,
                            "timestamp": timestamp,
                            "issue_details": issue
                        })
                    
                    # Save the frame if requested
                    if args.save_frames:
                        frame_filename = f"issue_frame_{frame_number:06d}_{timestamp.replace(':', '_')}.jpg"
                        frame_path = frames_dir / frame_filename
                        cv2.imwrite(str(frame_path), frame)
                
                # Update the progress bar
                progress_bar.update(1)
    
    # Save the safety report
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(safety_report, f, indent=2)
    
    # Print summary
    print(f"\nAnalysis complete!")
    print(f"Processed {len(frame_batches)} frames")
    print(f"Detected {issues_count} safety issues")
    if args.save_all_frames:
        print(f"All analyzed frames saved to: {all_frames_dir}")
    print(f"Safety report saved to: {output_path}")
    
    # Cleanup
    del video_processor


if __name__ == "__main__":
    main()
