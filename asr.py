import datetime
import os
import subprocess
from typing import Dict, List, Optional, Tuple, Set, Any

# Third-party imports
from dotenv import load_dotenv
import humanfriendly
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import webcolors
from docx import Document
from docx.shared import Pt, RGBColor
from fpdf import FPDF
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Annotation
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

generate_kwargs_asr = {

}

generate_kwargs_diarization = {
    # 'num_speakers': 2,
    # 'max_speakers': 5,
}

audio_in_path = "in/"
audio_out_path = "out/"
return_timestamps = True

save_type = "docx_pdf"

load_dotenv()

def load_audio(audio_path_folder: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load audio files from a directory using librosa.

    Args:
        audio_path_folder: Path to directory containing audio files

    Returns:
        Tuple containing:
            - List of dictionaries with keys "path", "sampling_rate", and "array"
            - List of audio file names

    Raises:
        FileNotFoundError: If the audio_path_folder doesn't exist
        RuntimeError: If no audio files are found in the directory
    """
    import librosa

    if not os.path.exists(audio_path_folder):
        raise FileNotFoundError(f"Audio directory not found: {audio_path_folder}")

    audio_samples = []
    names = []

    for file in os.listdir(audio_path_folder):
        audio_path = os.path.join(audio_path_folder, file)
        if os.path.isfile(audio_path):
            try:
                audio, sampling_rate = librosa.load(audio_path, sr=None)
                name = os.path.basename(audio_path)
                sample = {"path": name, "sampling_rate": sampling_rate, "array": audio}
                audio_samples.append(sample)
                names.append(name)
            except Exception as e:
                print(f"Error loading audio file {audio_path}: {e}")

    if not audio_samples:
        raise RuntimeError(f"No audio files could be loaded from {audio_path_folder}")

    return audio_samples, names


def write_to_file(
    audio_names: List[str], 
    results: List[Dict[str, Any]], 
    speaker_annotations: List[Annotation], 
    audio_in_path: str, 
    audio_out_path: str, 
    save_type: str, 
    return_timestamps: bool = False
) -> None:
    """
    Write audio transcription to files in specified formats.

    Args:
        audio_names: List of original audio filenames
        results: List of ASR results for each audio file
        speaker_annotations: List of speaker diarization annotations
        audio_in_path: Path to the directory containing input audio files
        audio_out_path: Path to save output files
        save_type: Type of file to save to (format: "docx", "pdf", or "docx_pdf")
        return_timestamps: Whether to include timestamps in the output

    Raises:
        ValueError: If an invalid save_type is provided
        FileNotFoundError: If input files cannot be found
        OSError: If output directory cannot be created
    """
    import shutil

    # Create timestamped output directory
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    output_dir = os.path.join(audio_out_path, date)
    os.makedirs(output_dir, exist_ok=True)

    # For each audio file, create output files with transcription
    for i, (sample, result, annotations) in enumerate(zip(audio_names, results, speaker_annotations)):
        try:
            # Copy original audio file to output directory
            source_path = os.path.join(audio_in_path, sample)
            if not os.path.exists(source_path):
                print(f"Warning: Source audio file not found: {source_path}")
                continue

            shutil.copy(source_path, output_dir)

            # Get base filename without extension
            base_filename = os.path.splitext(sample)[0]

            # Process each requested output format
            for doc_type in save_type.split("_"):
                if doc_type == "docx":
                    create_docx_transcript(sample, result, base_filename, date, output_dir)
                    if return_timestamps:
                        write_timestamps_to_docx(sample, result, annotations, date, output_dir)
                elif doc_type == "pdf":
                    create_pdf_transcript(sample, result, base_filename, date, output_dir)
                    if return_timestamps:
                        write_timestamps_to_pdf(sample, result, date, output_dir)
                else:
                    raise ValueError(f"Invalid save_type '{doc_type}'. Supported types: 'docx', 'pdf'")
        except Exception as e:
            print(f"Error processing file {sample}: {e}")


def create_docx_transcript(
    sample: str, 
    result: Dict[str, Any], 
    base_filename: str, 
    date: str, 
    output_dir: str
) -> None:
    """Create a Word document with the transcription."""
    doc = Document()

    # Set font size and style
    doc.styles['Normal'].font.size = Pt(12)
    doc.styles['Normal'].font.name = 'Arial'

    doc.add_heading(f"Transcription for {sample} - {date}", level=1)
    # Add a line break
    doc.add_paragraph("")
    doc.add_paragraph(result["text"])

    output_path = os.path.join(output_dir, f"transcription_{base_filename}.docx")
    doc.save(output_path)


def create_pdf_transcript(
    sample: str, 
    result: Dict[str, Any], 
    base_filename: str, 
    date: str, 
    output_dir: str
) -> None:
    """Create a PDF document with the transcription."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Transcription for {sample} - {date}", ln=True)
    # Add a line break
    pdf.ln(10)
    # Add the long text using multi_cell
    pdf.multi_cell(0, 10, result["text"])

    output_path = os.path.join(output_dir, f"transcription_{base_filename}.pdf")
    pdf.output(output_path)


def process_diarization(annotation: Annotation) -> Tuple[List[Any], List[str]]:
    """
    Process the diarization annotation to extract segments and speaker identifiers.

    Args:
        annotation: Annotation object from pyannote.audio containing speaker diarization results

    Returns:
        Tuple containing:
            - List of segment objects representing time intervals
            - List of speaker identifiers corresponding to each segment

    Raises:
        ValueError: If the annotation is empty or invalid
    """
    if not annotation or not hasattr(annotation, '_tracks') or not annotation._tracks:
        raise ValueError("Invalid or empty annotation object")

    segments = []
    speakers = []

    for segment, speaker in annotation._tracks.items():
        if speaker and len(speaker) > 0:
            segments.append(segment)
            speakers.append(list(speaker.values())[0])

    if not segments:
        raise ValueError("No segments found in the annotation")

    return segments, speakers


def generate_n_colors(speakers: Set[str]) -> Tuple[Dict[str, RGBColor], List[str]]:
    """
    Generate a distinct color for each speaker using a colormap.

    Args:
        speakers: Set of unique speaker identifiers

    Returns:
        Tuple containing:
            - Dictionary mapping each speaker to an RGBColor
            - List of color names corresponding to each color

    Raises:
        ValueError: If speakers set is empty
    """
    if not speakers:
        raise ValueError("No speakers provided to generate colors for")

    n = len(speakers)

    # Use tab10 colormap which is designed for categorical data and is colorblind-friendly
    cmap = plt.get_cmap('tab10')

    # Generate evenly spaced colors from the colormap
    colors = [cmap(i) for i in np.linspace(0, 1, max(n, 1))]

    # Convert colors to RGBColor objects for use with python-docx
    rgb_colors = [
        RGBColor(
            int(c[0] * 255), 
            int(c[1] * 255), 
            int(c[2] * 255)
        ) for c in colors
    ]

    # Get human-readable color names for reference
    color_tuples = [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colors]
    color_names = [closest_color_name(rgb) for rgb in color_tuples]

    # Create mapping from speakers to colors
    speaker_color_map = dict(zip(speakers, rgb_colors))

    return speaker_color_map, color_names


def closest_color_name(rgb_tuple: Tuple[int, int, int]) -> str:
    """
    Find the closest CSS3 color name for a given RGB tuple.

    Args:
        rgb_tuple: Tuple of (red, green, blue) values, each in range 0-255

    Returns:
        String containing the CSS3 color name closest to the input RGB value

    Raises:
        TypeError: If rgb_tuple is not a tuple of 3 integers
    """
    # Validate input
    if not isinstance(rgb_tuple, tuple) or len(rgb_tuple) != 3:
        raise TypeError("rgb_tuple must be a tuple of 3 integers")

    if not all(isinstance(c, int) and 0 <= c <= 255 for c in rgb_tuple):
        raise ValueError("RGB values must be integers between 0 and 255")

    try:
        # Try to get the exact color name if it exists
        return webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        # If exact name is not found, find the closest color
        closest_name = None
        min_diff = float('inf')

        for name in webcolors.names("css3"):
            r, g, b = webcolors.name_to_rgb(name)
            # Calculate Euclidean distance in RGB space
            diff = ((r - rgb_tuple[0]) ** 2 +
                    (g - rgb_tuple[1]) ** 2 +
                    (b - rgb_tuple[2]) ** 2)

            if diff < min_diff:
                min_diff = diff
                closest_name = name

        return closest_name or "unknown"


def write_timestamps_to_docx(
    sample: str, 
    result: Dict[str, Any], 
    annotations: Annotation, 
    date: str, 
    write_path: str
) -> None:
    """
    Create a Word document with timestamped transcription and speaker information.

    Args:
        sample: Audio filename
        result: ASR result containing text and timestamps
        annotations: Speaker diarization annotations
        date: Date string for the document header
        write_path: Directory to save the document

    Raises:
        ValueError: If result doesn't contain required fields or annotations are invalid
        OSError: If the document cannot be saved
    """
    # Validate input
    if not isinstance(result, dict) or "chunks" not in result:
        raise ValueError("Result must be a dictionary containing 'chunks'")

    # Process diarization to get speaker information
    try:
        segments, speakers = process_diarization(annotations)
    except ValueError as e:
        print(f"Warning: Diarization processing failed: {e}")
        segments, speakers = [], []

    # Get unique speakers
    unique_speakers = set(speakers)

    # Generate colors for each speaker
    try:
        color_map, color_names = generate_n_colors(unique_speakers)
    except ValueError as e:
        print(f"Warning: Color generation failed: {e}")
        color_map, color_names = {}, []

    # Create a new document
    doc = Document()

    # Set font size and style
    doc.styles['Normal'].font.size = Pt(12)
    doc.styles['Normal'].font.name = 'Arial'

    # Add document header
    doc.add_heading(f"Transcription with timestamps for {sample} - {date}", level=1)

    # Add description if we have segments and speakers
    if segments and speakers:
        try:
            description = generate_description(result, segments, speakers)
            doc.add_paragraph(description, style='Heading 3')
        except Exception as e:
            print(f"Warning: Failed to generate description: {e}")

    # Introduce speakers with appropriate colors
    if color_map:
        for speaker, color in color_map.items():
            try:
                speaker_para = doc.add_paragraph(f"Speaker {speaker}", style='Heading 3')
                speaker_para.runs[0].font.color.rgb = color
            except Exception as e:
                print(f"Warning: Failed to add speaker color for {speaker}: {e}")

    # Add a line break
    doc.add_paragraph("")

    # Add each chunk with timestamp and speaker information
    for chunk in result.get("chunks", []):
        try:
            curr_text = chunk.get("text", "")
            times = chunk.get("timestamp", (0, 0))

            # Handle case where end time is None
            if times[1] is None:
                times = (times[0], times[0] + 1)

            # Format times
            start_time = f"{times[0]:.2f} s"
            end_time = f"{times[1]:.2f} s"

            # Find speaker for this time segment
            middle_s = (times[0] + times[1]) / 2
            speaker = get_speaker_of_chunk(middle_s, annotations)

            # Add paragraph with appropriate color if speaker is identified
            if speaker is not None and speaker in color_map:
                para = doc.add_paragraph(f"{start_time} - {end_time} ({speaker}): {curr_text}")
                para.runs[0].font.color.rgb = color_map[speaker]
            else:
                doc.add_paragraph(f"{start_time} - {end_time}: {curr_text}")
        except Exception as e:
            print(f"Warning: Failed to process chunk: {e}")

    # Save the document
    base_filename = os.path.splitext(sample)[0]
    output_path = os.path.join(write_path, f"transcription_timestamped_{base_filename}.docx")

    try:
        doc.save(output_path)
    except Exception as e:
        raise OSError(f"Failed to save document to {output_path}: {e}")


def get_speaker_of_chunk(middle_s: float, annotations: Annotation) -> Optional[str]:
    """
    Determine which speaker was active at a specific time point.

    Args:
        middle_s: Time point in seconds to check for speaker
        annotations: Speaker diarization annotations

    Returns:
        Speaker identifier if found, None otherwise
    """
    if not annotations or not hasattr(annotations, '_tracks'):
        return None

    try:
        # First check: direct overlap with a segment
        for segment, speaker in annotations._tracks.items():
            if segment.overlaps(middle_s) and speaker:
                return list(speaker.values())[0]

        # Second check: time point is between segments (pause)
        # Get all segments as a sorted list
        track_items = list(annotations._tracks.items())

        # Check pairs of consecutive segments
        for i in range(len(track_items) - 1):
            segment_first, speaker_first = track_items[i]
            segment_second, speaker_second = track_items[i + 1]

            # If the time point is between these segments
            if segment_first.end <= middle_s <= segment_second.start:
                # If both segments have the same speaker, use that speaker
                if speaker_first == speaker_second and speaker_first:
                    return list(speaker_first.values())[0]

                # Otherwise, use the speaker of the closest segment
                if speaker_first and speaker_second:
                    if middle_s - segment_first.end < segment_second.start - middle_s:
                        return list(speaker_first.values())[0]
                    else:
                        return list(speaker_second.values())[0]
                elif speaker_first:
                    return list(speaker_first.values())[0]
                elif speaker_second:
                    return list(speaker_second.values())[0]
    except Exception as e:
        print(f"Warning: Error determining speaker at time {middle_s}: {e}")

    return None


def generate_description(
    result: Dict[str, Any], 
    segments: List[Any], 
    speakers: List[str]
) -> str:
    """
    Generate a human-readable description of the transcription.

    Args:
        result: ASR result containing text chunks
        segments: List of time segments from diarization
        speakers: List of speaker identifiers

    Returns:
        A string describing the transcription length, word count, and speaker count

    Raises:
        ValueError: If input data is invalid or empty
        IndexError: If segments list is empty
    """
    if not segments:
        raise ValueError("Cannot generate description: No segments provided")

    if not result or "chunks" not in result:
        raise ValueError("Cannot generate description: Invalid result format")

    try:
        # Get the total duration from the last segment's end time
        length = humanfriendly.format_timespan(segments[-1].end)

        # Count total words in all chunks
        num_words = sum(len(chunk.get("text", "").split()) for chunk in result.get("chunks", []))

        # Count unique speakers
        n_speakers = len(set(speakers)) if speakers else 0

        # Create the description
        description = (
            f"Transcription is {length} long, and contains {num_words} words "
            f"spoken by {n_speakers} different speaker{'s' if n_speakers != 1 else ''}."
        )

        return description
    except Exception as e:
        # Provide a basic description if there's an error
        return f"Transcription processing completed. Error generating detailed stats: {e}"


def write_timestamps_to_pdf(
    sample: str, 
    result: Dict[str, Any], 
    date: str, 
    write_path: str
) -> None:
    """
    Create a PDF document with timestamped transcription.

    Args:
        sample: Audio filename
        result: ASR result containing text and timestamps
        date: Date string for the document header
        write_path: Directory to save the document

    Raises:
        ValueError: If result doesn't contain required fields
        OSError: If the document cannot be saved
    """
    # Validate input
    if not isinstance(result, dict) or "chunks" not in result:
        raise ValueError("Result must be a dictionary containing 'chunks'")

    try:
        # Create PDF document
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Add document header
        pdf.cell(200, 10, f"Transcription with timestamps for {sample} - {date}", ln=True)
        pdf.ln(5)  # Add some space after the header

        # Add each chunk with timestamp
        for chunk in result.get("chunks", []):
            try:
                curr_text = chunk.get("text", "")
                times = chunk.get("timestamp", (0, 0))

                # Handle case where end time is None
                if times[1] is None:
                    times = (times[0], times[0] + 1)

                # Format times
                start_time = f"{times[0]:.2f} s"
                end_time = f"{times[1]:.2f} s"

                # Add the text with timestamps
                pdf.multi_cell(0, 10, f"{start_time} - {end_time}: {curr_text}")
            except Exception as e:
                print(f"Warning: Failed to process chunk in PDF: {e}")
                continue

        # Save the document
        base_filename = os.path.splitext(sample)[0]
        output_path = os.path.join(write_path, f"transcription_timestamped_{base_filename}.pdf")

        pdf.output(output_path)
    except Exception as e:
        raise OSError(f"Failed to create or save PDF document: {e}")


def asr(
    sample: Dict[str, Any], 
    return_timestamps: bool, 
    generate_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform Automatic Speech Recognition (ASR) using Hugging Face's Transformers pipeline.

    Args:
        sample: Dictionary containing audio data with keys "path", "sampling_rate", and "array"
        return_timestamps: Whether to include word/segment timestamps in the result
        generate_kwargs: Optional parameters to pass to the generation step

    Returns:
        Dictionary containing transcription results

    Raises:
        ValueError: If sample doesn't contain required audio data
        RuntimeError: If model loading or inference fails
    """
    # Validate input
    if not isinstance(sample, dict) or "array" not in sample or "sampling_rate" not in sample:
        raise ValueError("Sample must be a dictionary with 'array' and 'sampling_rate' keys")

    try:
        # Set up device and precision
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Model configuration
        model_id = "openai/whisper-large-v3"

        # Load model with appropriate settings
        print(f"Loading ASR model {model_id} on {device}...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        model.to(device)

        # Load processor (tokenizer and feature extractor)
        processor = AutoProcessor.from_pretrained(model_id)

        # Create pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            batch_size=64,
            torch_dtype=torch_dtype,
            device=device,
        )

        # Process audio and get transcription
        print(f"Transcribing audio ({len(sample['array'])/sample['sampling_rate']:.2f} seconds)...")
        result = pipe(
            sample, 
            return_timestamps=return_timestamps, 
            generate_kwargs=generate_kwargs or {}
        )

        return result
    except Exception as e:
        raise RuntimeError(f"ASR processing failed: {e}")


def diarization(audio_folder: str, **kwargs) -> List[Annotation]:
    """
    Perform speaker diarization on audio files using pyannote.audio.

    Args:
        audio_folder: Path to directory containing audio files
        **kwargs: Additional keyword arguments for the diarization pipeline
                 (e.g., num_speakers, min_speakers, max_speakers)

    Returns:
        List of Annotation objects containing speaker diarization results for each file

    Raises:
        ValueError: If audio_folder doesn't exist or contains no audio files
        RuntimeError: If diarization fails or authentication token is missing
    """
    # Validate input directory
    if not os.path.exists(audio_folder):
        raise ValueError(f"Audio folder not found: {audio_folder}")

    # Check for Hugging Face token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN environment variable not set. "
            "This is required for accessing the speaker diarization model."
        )

    try:
        # Set up device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load the diarization pipeline
        print(f"Loading speaker diarization model on {device}...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        pipeline.to(device)

        # Get list of audio files
        audio_files = [f for f in os.listdir(audio_folder) 
                      if os.path.isfile(os.path.join(audio_folder, f)) and 
                      f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]

        if not audio_files:
            raise ValueError(f"No audio files found in {audio_folder}")

        annotations = []

        # Process each audio file
        for i, file in enumerate(audio_files):
            try:
                audio_path = os.path.join(audio_folder, file)
                print(f"Processing diarization for file {i+1}/{len(audio_files)}: {file}")

                # Load audio using torchaudio for better performance
                waveform, sample_rate = torchaudio.load(audio_path)
                audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}

                # Run the pipeline with progress tracking
                with ProgressHook() as hook:
                    annotation = pipeline(audio_in_memory, hook=hook, **kwargs)

                annotations.append(annotation)
                print(f"Identified {len(set(annotation.labels()))} speakers in {file}")

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                # Add None to maintain alignment with input files
                annotations.append(None)

        # Check if any files were successfully processed
        if not any(annotations):
            raise RuntimeError("Failed to process any audio files")

        return [a for a in annotations if a is not None]

    except Exception as e:
        raise RuntimeError(f"Speaker diarization failed: {e}")


def parse_arguments():
    """Parse command line arguments for the ASR script."""
    import argparse

    parser = argparse.ArgumentParser(description="Automatic Speech Recognition and Speaker Diarization")

    parser.add_argument(
        "--input", "-i", 
        type=str, 
        default="in/",
        help="Path to input directory containing audio files (default: 'in/')"
    )

    parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="out/",
        help="Path to output directory for transcriptions (default: 'out/')"
    )

    parser.add_argument(
        "--format", "-f", 
        type=str, 
        default="docx_pdf",
        choices=["docx", "pdf", "docx_pdf"],
        help="Output format(s) for transcriptions (default: 'docx_pdf')"
    )

    parser.add_argument(
        "--timestamps", "-t", 
        action="store_true", 
        default=True,
        help="Include timestamps in transcription (default: True)"
    )

    parser.add_argument(
        "--no-timestamps", 
        action="store_false", 
        dest="timestamps",
        help="Disable timestamps in transcription"
    )

    parser.add_argument(
        "--speakers", "-s", 
        type=int, 
        help="Specify the number of speakers (optional)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Configure settings from arguments
        audio_in_path = args.input
        audio_out_path = args.output
        save_type = args.format
        return_timestamps = args.timestamps

        # Set up diarization kwargs if speakers specified
        generate_kwargs_diarization = {}
        if args.speakers:
            generate_kwargs_diarization["num_speakers"] = args.speakers

        # Start timer
        start = datetime.datetime.now()
        print(f"Starting ASR process at {start.strftime('%Y-%m-%d %H:%M:%S')}")

        # Load audio files
        print(f"Loading audio from {audio_in_path}...")
        audio_samples, names = load_audio(audio_in_path)
        print(f"Loaded {len(audio_samples)} audio files")

        # Run ASR on each audio file
        print("Running Automatic Speech Recognition...")
        results = []
        for i, sample in enumerate(audio_samples):
            print(f"Processing ASR for file {i+1}/{len(audio_samples)}: {sample['path']}")
            result = asr(sample, return_timestamps=return_timestamps, generate_kwargs=generate_kwargs_asr)
            results.append(result)

        # Run speaker diarization if timestamps are enabled
        speaker_annotations = []
        if return_timestamps:
            print("Running Speaker Diarization...")
            speaker_annotations = diarization(audio_in_path, **generate_kwargs_diarization)

        # Write results to files
        print(f"Writing transcriptions to {audio_out_path}...")
        write_to_file(
            names, 
            results, 
            speaker_annotations, 
            audio_in_path, 
            audio_out_path, 
            save_type, 
            return_timestamps=return_timestamps
        )

        # End timer and report
        end = datetime.datetime.now()
        duration = end - start
        print(f"Process completed at {end.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time taken: {duration}")
        print(f"Successfully processed {len(results)} files")
        print("Done!")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
