import sys, os
sys.path.append(os.getcwd())

import typer
import cv2
from pathlib import Path
from typing import Optional

from utils.file_sys import ensure_path_exists
from utils.concurrent import AsyncWorkerPool


def extract_frames_with_config(
    video_path: Path,
    output_dir: Path,
    target_fps: Optional[float] = None,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    start_index: int = 0
):
    """
    Extract frames from a video with optional resolution and framerate control.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        target_fps: Target framerate for extraction (None = use all frames)
        target_width: Target width (None = keep original or calculate from height)
        target_height: Target height (None = keep original or calculate from width)
        start_index: Starting index for frame numbering
    """
    ensure_path_exists(output_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate target resolution maintaining aspect ratio
    resolution = None
    if target_width is not None or target_height is not None:
        aspect_ratio = original_width / original_height

        if target_width is not None and target_height is not None:
            # Both specified, use as-is
            resolution = (target_width, target_height)
        elif target_width is not None:
            # Only width specified, calculate height
            resolution = (target_width, int(target_width / aspect_ratio))
        else:
            # Only height specified, calculate width
            resolution = (int(target_height * aspect_ratio), target_height)

        print(f"Resizing from {original_width}x{original_height} to {resolution[0]}x{resolution[1]}")

    # Calculate frame sampling rate
    if target_fps is not None and target_fps > 0:
        if target_fps > original_fps:
            print(f"Warning: Target FPS ({target_fps}) is higher than original FPS ({original_fps:.2f}), using original FPS")
            frame_interval = 1
        else:
            frame_interval = int(original_fps / target_fps)
    else:
        frame_interval = 1

    frame_idx = 0
    saved_count = 0

    print(f"Processing {video_path.name} (FPS: {original_fps:.2f}, Total frames: {total_frames})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample frames based on target FPS
        if frame_idx % frame_interval == 0:
            # Resize if resolution is specified
            if resolution is not None:
                frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_LANCZOS4)

            # Save frame
            frame_path = output_dir / f"{start_index + saved_count:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_path.name}")
    return saved_count


def process_videos(
    input_path: str = typer.Argument(..., help="Path to video file or directory containing videos"),
    output_dir: Optional[str] = typer.Argument(None, help="Output directory for extracted frames. Defaults to folder with same name as input."),
    fps: Optional[float] = typer.Option(None, help="Target framerate for extraction (frames per second). None means extract all frames."),
    width: Optional[int] = typer.Option(None, help="Target width in pixels. Height will be calculated to maintain aspect ratio if not specified."),
    height: Optional[int] = typer.Option(None, help="Target height in pixels. Width will be calculated to maintain aspect ratio if not specified."),
    video_extensions: str = typer.Option("mp4,avi,mov,mkv,MP4,AVI,MOV,MKV", help="Comma-separated list of video file extensions to process"),
    worker_num: int = typer.Option(4, help="Number of parallel workers for processing multiple videos"),
    preserve_structure: bool = typer.Option(False, help="When input is a directory, preserve directory structure in output"),
):
    """
    Decompose videos into frames with configurable resolution and framerate.

    Examples:
        # Extract all frames to default output folder (video/)
        python scripts/dataset/video_to_frames.py video.mp4

        # Extract all frames from a single video at original resolution and FPS
        python scripts/dataset/video_to_frames.py video.mp4 output/frames

        # Extract frames at 10 FPS
        python scripts/dataset/video_to_frames.py video.mp4 --fps 10

        # Resize to width 640, height calculated automatically to maintain aspect ratio
        python scripts/dataset/video_to_frames.py video.mp4 --width 640

        # Resize to height 480, width calculated automatically to maintain aspect ratio
        python scripts/dataset/video_to_frames.py video.mp4 --height 480

        # Resize to exact 640x480 (may distort if aspect ratio doesn't match)
        python scripts/dataset/video_to_frames.py video.mp4 output/frames --width 640 --height 480

        # Process all videos in a directory with 5 FPS and width 1920
        python scripts/dataset/video_to_frames.py videos/ output/frames --fps 5 --width 1920

        # Process directory with 8 parallel workers (outputs to videos_frames/ by default)
        python scripts/dataset/video_to_frames.py videos/ --worker-num 8
    """
    input_path = Path(input_path)

    # Set default output_dir if not provided
    if output_dir is None:
        if input_path.is_file():
            # For a file like "video.mp4", output to "video/"
            output_dir = input_path.parent / input_path.stem
        else:
            # For a directory like "videos/", output to "videos_frames/"
            output_dir = input_path.parent / f"{input_path.name}_frames"

    output_dir = Path(output_dir)

    # Validate resolution parameters
    if width is not None and width <= 0:
        raise typer.BadParameter("Width must be a positive integer")
    if height is not None and height <= 0:
        raise typer.BadParameter("Height must be a positive integer")

    # Validate FPS
    if fps is not None and fps <= 0:
        raise typer.BadParameter("FPS must be a positive number")

    # Parse video extensions
    extensions = [f".{ext.strip().lower()}" for ext in video_extensions.split(",")]

    # Collect video files to process
    video_files = []
    if input_path.is_file():
        if input_path.suffix.lower() in extensions:
            video_files.append(input_path)
        else:
            print(f"Error: {input_path} is not a recognized video file")
            raise typer.Exit(1)
    elif input_path.is_dir():
        for video_path in input_path.rglob("*"):
            if video_path.is_file() and video_path.suffix.lower() in extensions:
                video_files.append(video_path)
        if not video_files:
            print(f"Error: No video files found in {input_path}")
            raise typer.Exit(1)
    else:
        print(f"Error: {input_path} does not exist")
        raise typer.Exit(1)

    print(f"Found {len(video_files)} video(s) to process")
    res_str = 'original'
    if width is not None and height is not None:
        res_str = f"{width}x{height}"
    elif width is not None:
        res_str = f"{width}x? (maintaining aspect ratio)"
    elif height is not None:
        res_str = f"?x{height} (maintaining aspect ratio)"
    print(f"Settings: FPS={fps if fps else 'original'}, Resolution={res_str}")

    # Process videos
    if len(video_files) == 1:
        # Single video: output directly to output_dir
        video_path = video_files[0]
        ensure_path_exists(output_dir)
        extract_frames_with_config(video_path, output_dir, fps, width, height)
    else:
        # Multiple videos: create subdirectory for each video
        tasks = []
        for video_path in video_files:
            if preserve_structure and input_path.is_dir():
                # Preserve directory structure
                rel_path = video_path.relative_to(input_path)
                video_output_dir = output_dir / rel_path.parent / video_path.stem
            else:
                # Flat structure with video name as subdirectory
                video_output_dir = output_dir / video_path.stem

            tasks.append((video_path, video_output_dir, fps, width, height, 0))

        # Process in parallel
        pool = AsyncWorkerPool(worker_num)
        for args in tasks:
            pool.add_task(extract_frames_with_config, *args)
        pool.wait_for_results()

    print(f"\nDone! Frames saved to {output_dir}")


if __name__ == "__main__":
    typer.run(process_videos)
