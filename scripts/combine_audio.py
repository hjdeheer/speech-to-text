#!/usr/bin/env python3
"""
Audio Combiner Script

This script takes multiple audio files as command line arguments and combines
them into a single audio file using ffmpeg. Files are concatenated in the order
they are provided.
"""

import os
import sys
import argparse
import subprocess
import tempfile
from typing import List


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


def validate_audio_files(file_paths: List[str]) -> List[str]:
    """
    Validate that all provided files exist and are likely audio files.
    
    Args:
        file_paths: List of paths to audio files
        
    Returns:
        List of valid file paths
    """
    audio_extensions = ('.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma')
    valid_files = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        if not file_path.lower().endswith(audio_extensions):
            print(f"Warning: File may not be an audio file: {file_path}")
            
        valid_files.append(file_path)
    
    return valid_files


def create_concat_file(audio_files: List[str]) -> str:
    """
    Create a temporary file listing all audio files for ffmpeg concat.
    
    Args:
        audio_files: List of paths to audio files
        
    Returns:
        Path to temporary concat file
    """
    # Create temporary file for ffmpeg concat demuxer
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    
    try:
        for audio_file in audio_files:
            # Convert to absolute path to avoid concat demuxer issues
            abs_path = os.path.abspath(audio_file)
            # Escape single quotes and wrap in single quotes for ffmpeg
            escaped_path = abs_path.replace("'", "'\\''")
            temp_file.write(f"file '{escaped_path}'\n")
        
        temp_file.close()
        return temp_file.name
        
    except Exception as e:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise e


def combine_audio_files(
    audio_files: List[str], 
    output_path: str,
    audio_format: str = 'mp3',
    audio_bitrate: str = '192k'
) -> bool:
    """
    Combine multiple audio files into a single file using ffmpeg.
    
    Args:
        audio_files: List of paths to audio files to combine
        output_path: Path for the combined output file
        audio_format: Output audio format
        audio_bitrate: Audio bitrate for the output file
        
    Returns:
        True if combination was successful, False otherwise
    """
    if len(audio_files) < 2:
        print("Error: At least 2 audio files are required for combining")
        return False
    
    concat_file_path = None
    
    try:
        # Create concat file for ffmpeg
        concat_file_path = create_concat_file(audio_files)
        
        # Determine audio codec based on format
        if audio_format == 'mp3':
            codec = 'libmp3lame'
        elif audio_format == 'wav':
            codec = 'pcm_s16le'
        elif audio_format == 'ogg':
            codec = 'libvorbis'
        elif audio_format == 'm4a':
            codec = 'aac'
        else:
            codec = 'copy'  # Copy codec if unsure
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-f', 'concat',              # Use concat demuxer
            '-safe', '0',                # Allow unsafe file paths
            '-i', concat_file_path,      # Input concat file
            '-c:a', codec,               # Audio codec
        ]
        
        # Add bitrate if not copying codec
        if codec != 'copy':
            cmd.extend(['-b:a', audio_bitrate])
        
        cmd.extend([
            '-y',                        # Overwrite output file if it exists
            output_path
        ])
        
        print(f"Combining {len(audio_files)} audio files...")
        print(f"Output: {output_path}")
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if result.returncode == 0:
            print(f"Success: Combined audio saved to {output_path}")
            return True
        else:
            print(f"Error: ffmpeg combination failed with code {result.returncode}")
            print(f"Error details: {result.stderr.decode('utf-8')}")
            return False
            
    except Exception as e:
        print(f"Error during audio combination: {e}")
        return False
        
    finally:
        # Clean up temporary concat file
        if concat_file_path and os.path.exists(concat_file_path):
            os.unlink(concat_file_path)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Combine multiple audio files into a single file using ffmpeg",
        epilog="Example: python combine_audio.py file1.mp3 file2.wav file3.mp3 -o combined.mp3"
    )
    
    parser.add_argument(
        "audio_files",
        nargs="+",
        help="Input audio files to combine (minimum 2 files required)"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for the combined audio file"
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
        help="Audio bitrate for output file (default: 192k)"
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
    
    # Validate input files
    valid_files = validate_audio_files(args.audio_files)
    
    if len(valid_files) < 2:
        print("Error: At least 2 valid audio files are required")
        sys.exit(1)
    
    # Print files to be combined
    print("Files to combine:")
    for i, file_path in enumerate(valid_files, 1):
        print(f"  {i}. {file_path}")
    
    # Combine audio files
    success = combine_audio_files(
        valid_files,
        args.output,
        args.format,
        args.bitrate
    )
    
    if success:
        print("Audio combination completed successfully!")
        sys.exit(0)
    else:
        print("Audio combination failed")
        sys.exit(1)