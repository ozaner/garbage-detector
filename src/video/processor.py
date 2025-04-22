"""
Video processor module for capturing, processing, and extracting frames from video.
"""
import cv2
import numpy as np
from pathlib import Path


class VideoProcessor:
    """
    A class to handle video capture, processing, and frame extraction.
    """
    
    def __init__(self, video_path):
        """
        Initialize the video processor.
        
        Args:
            video_path (str or Path): Path to the video file
        
        Raises:
            FileNotFoundError: If the video file does not exist
            ValueError: If the video file cannot be opened
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0
    
    def __del__(self):
        """Release the video capture when the object is deleted."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
    
    def get_frame(self, frame_number=None):
        """
        Get a specific frame or the next frame from the video.
        
        Args:
            frame_number (int, optional): The specific frame number to retrieve.
                If None, the next frame is retrieved.
        
        Returns:
            tuple: (success, frame) where success is a boolean indicating if the
                   frame was successfully retrieved and frame is the image data
        """
        if frame_number is not None:
            if frame_number < 0 or frame_number >= self.frame_count:
                return False, None
            
            # Set the position to the requested frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number
        
        success, frame = self.cap.read()
        if success:
            self.current_frame += 1
        return success, frame
    
    def get_frame_at_intervals(self, interval):
        """
        Generator that yields frames at specified intervals.
        
        Args:
            interval (int): Number of frames to skip between each yielded frame
        
        Yields:
            tuple: (frame_number, frame) where frame_number is the current frame number
                  and frame is the image data
        """
        # Reset to beginning of video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame = 0
        
        frame_number = 0
        while frame_number < self.frame_count:
            success, frame = self.cap.read()
            if not success:
                break
                
            if frame_number % interval == 0:
                yield frame_number, frame
            
            frame_number += 1
    
    def get_frame_timestamp(self, frame_number):
        """
        Convert a frame number to a timestamp.
        
        Args:
            frame_number (int): The frame number
        
        Returns:
            str: The timestamp in the format 'HH:MM:SS'
        """
        if frame_number < 0 or frame_number >= self.frame_count:
            raise ValueError(f"Frame number {frame_number} is out of bounds")
        
        seconds = frame_number / self.fps
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    
    def frames_to_video_position(self, frame_number):
        """
        Convert a frame number to a position in the video (percentage).
        
        Args:
            frame_number (int): The frame number
        
        Returns:
            float: The position in the video as a percentage (0-100)
        """
        if frame_number < 0 or frame_number >= self.frame_count:
            raise ValueError(f"Frame number {frame_number} is out of bounds")
        
        return (frame_number / self.frame_count) * 100
    
    def save_frame(self, frame, output_path):
        """
        Save a frame to disk.
        
        Args:
            frame (numpy.ndarray): The frame to save
            output_path (str or Path): The path to save the frame to
        
        Returns:
            bool: True if the frame was successfully saved, False otherwise
        """
        return cv2.imwrite(str(output_path), frame) 