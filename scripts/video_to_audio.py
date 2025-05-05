#!/usr/bin/env python3
"""
Video to Audio Converter

This script converts video files to audio files using ffmpeg.
It can process a single file or all video files in a directory.
"""

import os
import sys
import argparse
import subprocess
from typing import List, Optional


def check_ffmpeg_installed() -> bool:
    """Check if ffmpeg is installed on the system."""
    try:
        subprocess.run(
            ['ffmpeg', '-version'], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        return True
    except FileNotFoundError:
        return False


def convert_video_to_audio(
    video_path: str, 
    output_path: Optional[str] = None, 
    audio_format: str = 'mp3',
    audio_bitrate: str = '192k'
) -> bool:
    """
    Convert a video file to an audio file using ffmpeg.

    Args:
        video_path: Path to the video file
        output_path: Path to save the audio file (if None, uses the same location as video)
        audio_format: Output audio format (mp3, wav, etc.)
        audio_bitrate: Audio bitrate for the output file

    Returns:
        True if conversion was successful, False otherwise
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False

    # If output_path is not specified, create one based on the video path
    if output_path is None:
        video_dir = os.path.dirname(video_path)
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]
        output_path = os.path.join(video_dir, f"{video_name}.{audio_format}")

    try:
        # Run ffmpeg to convert the video to audio
        cmd = [
            'ffmpeg', 
            '-i', video_path,             # Input file
            '-vn',                        # Disable video
            '-acodec', 'libmp3lame' if audio_format == 'mp3' else 'pcm_s16le',  # Audio codec
            '-ab', audio_bitrate,         # Audio bitrate
            '-ar', '44100',               # Audio sample rate
            '-y',                         # Overwrite output file if it exists
            output_path
        ]

        print(f"Converting {video_path} to {output_path}...")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if result.returncode == 0:
            print(f"Success: Audio saved to {output_path}")
            return True
        else:
            print(f"Error: ffmpeg conversion failed with code {result.returncode}")
            print(f"Error details: {result.stderr.decode('utf-8')}")
            return False

    except Exception as e:
        print(f"Error during conversion: {e}")
        return False


def process_directory(
    input_dir: str, 
    output_dir: Optional[str] = None,
    audio_format: str = 'mp3',
    audio_bitrate: str = '192k'
) -> List[str]:
    """
    Process all video files in a directory.

    Args:
        input_dir: Directory containing video files
        output_dir: Directory to save audio files (if None, uses input_dir)
        audio_format: Output audio format
        audio_bitrate: Audio bitrate for output files

    Returns:
        List of paths to successfully created audio files
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return []

    if output_dir is not None and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory: {e}")
            return []

    # Get all video files in the directory
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
    video_files = [
        f for f in os.listdir(input_dir) 
        if os.path.isfile(os.path.join(input_dir, f)) and 
        f.lower().endswith(video_extensions)
    ]

    if not video_files:
        print(f"No video files found in {input_dir}")
        return []

    print(f"Found {len(video_files)} video files to process")

    successful_conversions = []

    for i, video_file in enumerate(video_files):
        video_path = os.path.join(input_dir, video_file)

        if output_dir is not None:
            # Create output path in the specified output directory
            video_name = os.path.splitext(video_file)[0]
            output_path = os.path.join(output_dir, f"{video_name}.{audio_format}")
        else:
            output_path = None  # Let convert_video_to_audio determine the path

        print(f"Processing file {i+1}/{len(video_files)}: {video_file}")
        success = convert_video_to_audio(
            video_path, 
            output_path, 
            audio_format, 
            audio_bitrate
        )

        if success and output_path:
            successful_conversions.append(output_path)
        elif success:
            # If output_path was None, construct it to return
            video_name = os.path.splitext(video_file)[0]
            auto_output_path = os.path.join(
                input_dir if output_dir is None else output_dir,
                f"{video_name}.{audio_format}"
            )
            successful_conversions.append(auto_output_path)

    return successful_conversions


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert video files to audio using ffmpeg"
    )

    parser.add_argument(
        "input",
        help="Input video file or directory containing video files"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output audio file or directory (default: same as input)"
    )

    parser.add_argument(
        "--format", "-f",
        default="mp3",
        choices=["mp3", "wav", "ogg", "m4a"],
        help="Output audio format (default: mp3)"
    )

    parser.add_argument(
        "--bitrate", "-b",
        default="192k",
        help="Audio bitrate (default: 192k)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Check if ffmpeg is installed
    if not check_ffmpeg_installed():
        print("Error: ffmpeg is not installed or not in PATH")
        print("Please install ffmpeg to use this script")
        sys.exit(1)

    # Parse command line arguments
    args = parse_arguments()

    # Determine if input is a file or directory
    input_path = args.input
    output_path = args.output
    audio_format = args.format
    audio_bitrate = args.bitrate

    if os.path.isfile(input_path):
        # Process a single file
        success = convert_video_to_audio(
            input_path, 
            output_path, 
            audio_format, 
            audio_bitrate
        )
        sys.exit(0 if success else 1)
    elif os.path.isdir(input_path):
        # Process all video files in the directory
        successful_files = process_directory(
            input_path, 
            output_path, 
            audio_format, 
            audio_bitrate
        )

        if successful_files:
            print(f"Successfully converted {len(successful_files)} files")
            sys.exit(0)
        else:
            print("No files were successfully converted")
            sys.exit(1)
    else:
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
