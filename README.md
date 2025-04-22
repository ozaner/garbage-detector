# Garbage Detector

A Python application that analyzes video footage from a garbage truck's point of view to detect potential safety issues when collecting trash.

## Features

- Analyzes video footage from garbage truck's POV
- Detects safety issues in trash collection (fire, hazardous materials, etc.)
- Customizable frame interval for analysis
- Generates JSON reports of detected safety issues
- Option to save frames with detected issues for review
- Parallel processing for improved performance

## Installation

### Clone this repository:
```bash
git clone https://github.com/yourusername/garbage-detector.git
cd garbage-detector
```

### Install dependencies:

When using `uv` there is no need! To install `uv` just run:
```bash
#macos
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
#windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then restart your terminal.

### Set up your OpenAI API key:
```bash
# Copy the example .env file
cp .env.example .env
```
Edit the .env file and add your API key


## Usage

```bash
# Help
uv run main.py --help

# Basic usage
uv run main.py path/to/video.mp4

# Analyze every 10th frame and save any frame with a detected issue
uv run main.py path/to/video.mp4 -n 10 --save-frames

# Save all analyzed frames and use 8 worker threads
uv run main.py path/to/video.mp4 --save-all-frames --workers 8
```

### Command Line Arguments

- `video_path`: Path to the video file to analyze
- `-o, --output`: Path to save the safety report (default: safety_report.json)
- `-n, --frame-interval`: Analyze every n-th frame (default: 30)
- `--save-frames`: Save frames with detected safety issues
- `--frames-dir`: Directory to save frames with detected issues (default: detected_frames)
- `--save-all-frames`: Save all analyzed frames regardless of safety issues
- `--all-frames-dir`: Directory to save all analyzed frames (default: all_frames)
- `--workers`: Number of parallel workers for frame analysis (default: 4)
- `--batch-size`: Number of frames to process in each batch (default: 10)

## Output

The application generates a JSON report with the following structure:

```json
{
  "video_file": "path/to/your/video.mp4",
  "analysis_timestamp": "2023-06-21 15:30:45",
  "frame_interval": 30,
  "detected_issues": [
    {
      "frame_number": 150,
      "timestamp": "00:05:00",
      "issue_details": {
        "issue_type": "fire",
        "location": "left trash can",
        "description": "Visible flames coming from the trash can on the left side of the frame"
      }
    }
  ]
}
```

