# /// script
# dependencies = [
#   "simple_term_menu",
#   "numpy",
#   "opencv-python"
# ]
# ///

"""
Cost calculator for estimating expenses of running AI models on video processing.

This script calculates the approximate cost for analyzing video frames 
with the garbage truck safety detection system (main.py) using different 
AI models, video resolutions, and frame intervals.
"""
import argparse
import math
import cv2
import numpy as np
from simple_term_menu import TerminalMenu

# LLM API Pricing (as of April 28, 2025)
MODEL_PRICING = {
    "gpt-4.1": {
        "input_text": 0.002,    # $2.00 per 1M tokens = $0.002 per 1K tokens
        "output_text": 0.008,   # $8.00 per 1M tokens = $0.008 per 1K tokens
        "image_calc_method": "tile"
    },
    "gpt-4.1-mini": {
        "input_text": 0.0004,   # $0.40 per 1M tokens = $0.0004 per 1K tokens
        "output_text": 0.0016,  # $1.60 per 1M tokens = $0.0016 per 1K tokens
        "image_calc_method": "patch"
    },
    "gpt-4o": {
        "input_text": 0.005,    # $5.00 per 1M tokens = $0.005 per 1K tokens
        "output_text": 0.015,   # $15.00 per 1M tokens = $0.015 per 1K tokens
        "image_calc_method": "tile"
    },
    "gpt-4o-mini": {
        "input_text": 0.00015,  # $0.15 per 1M tokens = $0.00015 per 1K tokens
        "output_text": 0.0006,  # $0.60 per 1M tokens = $0.0006 per 1K tokens
        "image_calc_method": "tile_mini"
    },
    "gemini-2.5-pro": {
        "input_text": 0.00125,  # $1.25 per 1M tokens = $0.00125 per 1K tokens
        "output_text": 0.01,    # $10.00 per 1M tokens = $0.01 per 1K tokens
        "image_calc_method": "gemini"
    },
    "gemini-2.5-flash": {
        "input_text": 0.00015,  # $0.15 per 1M tokens = $0.00015 per 1K tokens
        "output_text": 0.0035,  # $3.50 per 1M tokens = $0.0035 per 1K tokens
        "image_calc_method": "gemini"
    },
    "claude-3.7-sonnet": {
        "input_text": 0.003,    # $3.00 per 1M tokens = $0.003 per 1K tokens
        "output_text": 0.015,   # $15.00 per 1M tokens = $0.015 per 1K tokens
        "image_calc_method": "pixel"
    }
}

# Constants for our specific use case
PROMPT_TOKENS = 155  # As specified in the requirements
AVG_OUTPUT_TOKENS_PER_FRAME = 50  # Placeholder estimate for output tokens per frame

# Available video resolutions (width, height)
RESOLUTIONS = {
    "240p (426x240)": (426, 240),
    "360p (640x360)": (640, 360),
    "480p (854x480)": (854, 480),
    "VGA (640x480)": (640, 480),
    "720p (1280x720)": (1280, 720),
    "1080p (1920x1080)": (1920, 1080),
    "1440p (2560x1440)": (2560, 1440),
    "4K (3840x2160)": (3840, 2160)
}

def calculate_frames(video_length_seconds, fps, frame_interval):
    """Calculate the number of frames that will be processed."""
    total_frames = video_length_seconds * fps
    processed_frames = math.ceil(total_frames / frame_interval)
    return processed_frames

def calculate_image_tokens_tile(width, height, detail="high"):
    """
    Calculate image tokens for GPT-4.1 and GPT-4o using tile-based method.
    """
    if detail == "low":
        return 85
    
    # Step 1: Scale image - longest side <= 2048px
    scale_factor = min(1.0, 2048 / max(width, height))
    scaled_width = int(width * scale_factor)
    scaled_height = int(height * scale_factor)
    
    # Step 2: Scale image again - shortest side <= 768px
    scale_factor_2 = min(1.0, 768 / min(scaled_width, scaled_height))
    final_width = int(scaled_width * scale_factor_2)
    final_height = int(scaled_height * scale_factor_2)
    
    # Step 3: Calculate num_tiles
    tiles_wide = math.ceil(final_width / 512)
    tiles_high = math.ceil(final_height / 512)
    num_tiles = tiles_wide * tiles_high
    
    # Step 4: Calculate tokens
    return 85 + (170 * num_tiles)

def calculate_image_tokens_patch(width, height):
    """
    Calculate image tokens for GPT-4.1-mini using patch-based method.
    """
    # Calculate needed 32x32 patches
    patches_wide = math.ceil(width / 32)
    patches_high = math.ceil(height / 32)
    total_patches_needed = patches_wide * patches_high
    
    # Apply patch limit
    max_patches = 1536
    if total_patches_needed > max_patches:
        # We need to scale down the image
        scale_factor = math.sqrt(max_patches / total_patches_needed)
        patches_wide = math.ceil(width * scale_factor / 32)
        patches_high = math.ceil(height * scale_factor / 32)
        image_tokens = patches_wide * patches_high
    else:
        image_tokens = total_patches_needed
    
    # Apply model-specific multiplier
    return image_tokens * 1.62

def calculate_image_tokens_tile_mini(width, height, detail="high"):
    """
    Calculate image tokens for GPT-4o mini using tile-based method with different values.
    """
    if detail == "low":
        return 85  # Using same as base tile method as a baseline
    
    # Same scaling as GPT-4.1/GPT-4o
    scale_factor = min(1.0, 2048 / max(width, height))
    scaled_width = int(width * scale_factor)
    scaled_height = int(height * scale_factor)
    
    scale_factor_2 = min(1.0, 768 / min(scaled_width, scaled_height))
    final_width = int(scaled_width * scale_factor_2)
    final_height = int(scaled_height * scale_factor_2)
    
    tiles_wide = math.ceil(final_width / 512)
    tiles_high = math.ceil(final_height / 512)
    num_tiles = tiles_wide * tiles_high
    
    # Different token values for GPT-4o mini
    return 2833 + (5667 * num_tiles)

def calculate_image_tokens_gemini(width, height):
    """
    Estimate image tokens for Gemini models using a simplified approach.
    """
    # Simplified estimation based on legacy Gemini logic
    if width <= 384 and height <= 384:
        return 258
    
    # For larger images, estimate using tiling
    scaled_width = min(width, 1024)
    scaled_height = min(height, 1024)
    
    # Estimate number of 384x384 tiles
    tiles_wide = math.ceil(scaled_width / 384)
    tiles_high = math.ceil(scaled_height / 384)
    num_tiles = tiles_wide * tiles_high
    
    return 258 * num_tiles

def calculate_image_tokens_pixel(width, height):
    """
    Calculate image tokens for Claude 3.7 Sonnet using pixel-based formula.
    """
    return int((width * height) / 750)

def calculate_image_tokens(model, width, height):
    """
    Calculate image tokens based on the model and image dimensions.
    """
    method = MODEL_PRICING[model]["image_calc_method"]
    
    if method == "tile":
        return calculate_image_tokens_tile(width, height)
    elif method == "patch":
        return calculate_image_tokens_patch(width, height)
    elif method == "tile_mini":
        return calculate_image_tokens_tile_mini(width, height)
    elif method == "gemini":
        return calculate_image_tokens_gemini(width, height)
    elif method == "pixel":
        return calculate_image_tokens_pixel(width, height)
    else:
        raise ValueError(f"Unknown image token calculation method: {method}")

def calculate_cost(model, frames, prompt_tokens, image_width, image_height):
    """Calculate the estimated cost for processing with a specific model."""
    # Calculate image tokens for each frame
    image_tokens_per_frame = calculate_image_tokens(model, image_width, image_height)
    total_image_tokens = image_tokens_per_frame * frames
    
    # Cost for image processing (uses input text rate)
    image_cost = (total_image_tokens / 1000) * MODEL_PRICING[model]["input_text"]
    
    # Cost for text processing (input and output)
    input_text_cost = (prompt_tokens * frames / 1000) * MODEL_PRICING[model]["input_text"]
    output_text_cost = (AVG_OUTPUT_TOKENS_PER_FRAME * frames / 1000) * MODEL_PRICING[model]["output_text"]
    
    total_cost = image_cost + input_text_cost + output_text_cost
    return {
        "image_tokens_per_frame": image_tokens_per_frame,
        "total_image_tokens": total_image_tokens,
        "image_cost": image_cost,
        "input_text_cost": input_text_cost,
        "output_text_cost": output_text_cost,
        "total_cost": total_cost
    }

def run_interactive_interface():
    """Run the interactive interface with arrow key navigation."""
    # Available options for selection
    models = list(MODEL_PRICING.keys())
    video_lengths = ["30 seconds", "1 minute", "5 minutes", "10 minutes", "30 minutes", "1 hour", "Custom..."]
    fps_options = ["24 fps", "30 fps", "60 fps", "Custom..."]
    frame_intervals = ["10", "30", "60", "120", "Custom..."]
    resolution_options = list(RESOLUTIONS.keys())
    
    # Title for the application
    print("\n=== AI Model Cost Calculator for Garbage Detector (main.py) ===\n")
    print("This tool estimates the cost of processing videos with different AI models.")
    print("Use arrow keys to navigate and Enter to select an option.\n")
    
    # Step 1: Select Model
    print("Step 1: Select an AI model:")
    model_menu = TerminalMenu(models, title="Available Models")
    model_choice = model_menu.show()
    
    if model_choice is None:
        print("Operation cancelled.")
        return
    
    selected_model = models[model_choice]
    print(f"Selected model: {selected_model}")
    
    # Step 2: Select Video Length
    print("\nStep 2: Select video length:")
    length_menu = TerminalMenu(video_lengths, title="Video Length")
    length_choice = length_menu.show()
    
    if length_choice is None:
        print("Operation cancelled.")
        return
    
    selected_length = video_lengths[length_choice]
    
    # Convert selection to seconds
    if selected_length == "Custom...":
        while True:
            try:
                minutes = float(input("Enter video length in minutes: "))
                video_length_seconds = minutes * 60
                break
            except ValueError:
                print("Please enter a valid number.")
    else:
        if selected_length == "30 seconds":
            video_length_seconds = 30
        elif selected_length == "1 minute":
            video_length_seconds = 60
        elif selected_length == "5 minutes":
            video_length_seconds = 300
        elif selected_length == "10 minutes":
            video_length_seconds = 600
        elif selected_length == "30 minutes":
            video_length_seconds = 1800
        elif selected_length == "1 hour":
            video_length_seconds = 3600
    
    print(f"Video length: {video_length_seconds} seconds")
    
    # Step 3: Select FPS
    print("\nStep 3: Select frames per second (FPS):")
    fps_menu = TerminalMenu(fps_options, title="Frames Per Second")
    fps_choice = fps_menu.show()
    
    if fps_choice is None:
        print("Operation cancelled.")
        return
    
    selected_fps = fps_options[fps_choice]
    
    # Convert selection to FPS value
    if selected_fps == "Custom...":
        while True:
            try:
                fps = int(input("Enter frames per second (FPS): "))
                break
            except ValueError:
                print("Please enter a valid integer.")
    else:
        fps = int(selected_fps.split()[0])
    
    print(f"Selected FPS: {fps}")
    
    # Step 4: Select Frame Interval
    print("\nStep 4: Select frame interval (analyze every n-th frame):")
    interval_menu = TerminalMenu(frame_intervals, title="Frame Interval")
    interval_choice = interval_menu.show()
    
    if interval_choice is None:
        print("Operation cancelled.")
        return
    
    selected_interval = frame_intervals[interval_choice]
    
    # Convert selection to interval value
    if selected_interval == "Custom...":
        while True:
            try:
                frame_interval = int(input("Enter frame interval: "))
                break
            except ValueError:
                print("Please enter a valid integer.")
    else:
        frame_interval = int(selected_interval)
    
    print(f"Selected frame interval: {frame_interval}")
    
    # Step 5: Select Video Resolution
    print("\nStep 5: Select video resolution:")
    res_menu = TerminalMenu(resolution_options, title="Video Resolution")
    res_choice = res_menu.show()
    
    if res_choice is None:
        print("Operation cancelled.")
        return
    
    selected_res = resolution_options[res_choice]
    width, height = RESOLUTIONS[selected_res]
    
    print(f"Selected resolution: {width}x{height}")
    
    # Calculate the results
    frames_to_process = calculate_frames(video_length_seconds, fps, frame_interval)
    cost_details = calculate_cost(selected_model, frames_to_process, PROMPT_TOKENS, width, height)
    
    # Display results
    print("\n=== Cost Estimation Results ===")
    print(f"Model: {selected_model}")
    print(f"Video length: {video_length_seconds} seconds ({video_length_seconds/60:.2f} minutes)")
    print(f"FPS: {fps}")
    print(f"Frame interval: Every {frame_interval} frames")
    print(f"Resolution: {width}x{height}")
    print(f"Total frames to process: {frames_to_process}")
    print("\nTokenization and cost details:")
    print(f"Image tokens per frame: {cost_details['image_tokens_per_frame']:.0f}")
    print(f"Total image tokens: {cost_details['total_image_tokens']:.0f}")
    print(f"Image processing cost: ${cost_details['image_cost']:.6f}")
    print(f"Input text processing cost: ${cost_details['input_text_cost']:.6f}")
    print(f"Output text processing cost: ${cost_details['output_text_cost']:.6f}")
    print(f"Total estimated cost: ${cost_details['total_cost']:.6f}")
    
    # Option to run another calculation
    print("\nWould you like to run another calculation?")
    options = ["Yes", "No"]
    another_menu = TerminalMenu(options)
    another_choice = another_menu.show()
    
    if another_choice == 0:  # "Yes"
        print("\n" + "="*50 + "\n")
        run_interactive_interface()

def compare_all_models():
    """Compare costs across all models for a given configuration."""
    # Get video parameters
    print("\n=== Compare All Models ===\n")
    
    # Video Length
    video_lengths = ["30 seconds", "1 minute", "5 minutes", "10 minutes", "30 minutes", "1 hour", "Custom..."]
    print("Select video length:")
    length_menu = TerminalMenu(video_lengths)
    length_choice = length_menu.show()
    
    if length_choice is None:
        print("Operation cancelled.")
        return
    
    selected_length = video_lengths[length_choice]
    
    # Convert selection to seconds
    if selected_length == "Custom...":
        while True:
            try:
                minutes = float(input("Enter video length in minutes: "))
                video_length_seconds = minutes * 60
                break
            except ValueError:
                print("Please enter a valid number.")
    else:
        if selected_length == "30 seconds":
            video_length_seconds = 30
        elif selected_length == "1 minute":
            video_length_seconds = 60
        elif selected_length == "5 minutes":
            video_length_seconds = 300
        elif selected_length == "10 minutes":
            video_length_seconds = 600
        elif selected_length == "30 minutes":
            video_length_seconds = 1800
        elif selected_length == "1 hour":
            video_length_seconds = 3600
    
    # FPS
    fps_options = ["24 fps", "30 fps", "60 fps", "Custom..."]
    print("\nSelect frames per second (FPS):")
    fps_menu = TerminalMenu(fps_options)
    fps_choice = fps_menu.show()
    
    if fps_choice is None:
        print("Operation cancelled.")
        return
    
    selected_fps = fps_options[fps_choice]
    
    # Convert selection to FPS value
    if selected_fps == "Custom...":
        while True:
            try:
                fps = int(input("Enter frames per second (FPS): "))
                break
            except ValueError:
                print("Please enter a valid integer.")
    else:
        fps = int(selected_fps.split()[0])
    
    # Frame Interval
    frame_intervals = ["10", "30", "60", "120", "Custom..."]
    print("\nSelect frame interval (analyze every n-th frame):")
    interval_menu = TerminalMenu(frame_intervals)
    interval_choice = interval_menu.show()
    
    if interval_choice is None:
        print("Operation cancelled.")
        return
    
    selected_interval = frame_intervals[interval_choice]
    
    # Convert selection to interval value
    if selected_interval == "Custom...":
        while True:
            try:
                frame_interval = int(input("Enter frame interval: "))
                break
            except ValueError:
                print("Please enter a valid integer.")
    else:
        frame_interval = int(selected_interval)
    
    # Video Resolution
    resolution_options = list(RESOLUTIONS.keys())
    
    print("\nSelect video resolution:")
    res_menu = TerminalMenu(resolution_options)
    res_choice = res_menu.show()
    
    if res_choice is None:
        print("Operation cancelled.")
        return
    
    selected_res = resolution_options[res_choice]
    width, height = RESOLUTIONS[selected_res]
    
    # Calculate frames to process
    frames_to_process = calculate_frames(video_length_seconds, fps, frame_interval)
    
    # Calculate and display costs for all models
    print("\n=== Cost Comparison for All Models ===")
    print(f"Video length: {video_length_seconds} seconds ({video_length_seconds/60:.2f} minutes)")
    print(f"FPS: {fps}")
    print(f"Frame interval: Every {frame_interval} frames")
    print(f"Resolution: {width}x{height}")
    print(f"Total frames to process: {frames_to_process}")
    print("\nEstimated costs by model:")
    print("-" * 100)
    header = f"{'Model':<20} {'Image Tokens/Frame':<20} {'Image Cost':<15} {'Text Cost':<15} {'Total Cost':<15}"
    print(header)
    print("-" * 100)
    
    for model in MODEL_PRICING:
        cost_details = calculate_cost(model, frames_to_process, PROMPT_TOKENS, width, height)
        total_text_cost = cost_details['input_text_cost'] + cost_details['output_text_cost']
        print(f"{model:<20} {cost_details['image_tokens_per_frame']:<20.0f} ${cost_details['image_cost']:<14.4f} ${total_text_cost:<14.4f} ${cost_details['total_cost']:<14.4f}")

def main():
    """Main function to run the cost calculator."""
    parser = argparse.ArgumentParser(description="Calculate AI model costs for video processing with main.py")
    parser.add_argument("--compare-all", action="store_true", help="Compare costs across all models")
    args = parser.parse_args()
    
    try:
        if args.compare_all:
            compare_all_models()
        else:
            # Main menu options
            options = [
                "Calculate Cost for a Single Model",
                "Compare All Models",
                "Exit"
            ]
            
            main_menu = TerminalMenu(options, title="AI Model Cost Calculator for main.py")
            menu_choice = main_menu.show()
            
            if menu_choice == 0:
                run_interactive_interface()
            elif menu_choice == 1:
                compare_all_models()
            else:
                print("Exiting...")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
