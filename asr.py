import datetime
# Create an folder inside audio_out_path to save the files based on the date, h m and seconds
import os

import torch
from docx import Document
from docx.shared import Pt
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

generate_kwargs = {

}

audio_in_path = "in/"
audio_out_path = "out/"
return_timestamps = True

save_type = "docx_pdf"


def load_audio(audio_path_folder: str) -> tuple[list[dict], list[str]]:
    """
    Load audio file from path using librosa.
    :param audio_path_folder: path to audio file
    :return: dictionary with keys "path", "sampling_rate", "array"
    """
    import librosa

    # Read all files inside the audio path folder using os listdir
    audio_samples = []
    names = []
    import os

    for file in os.listdir(audio_path_folder):
        audio_path = os.path.join(audio_path_folder, file)
        audio, sampling_rate = librosa.load(audio_path, sr=None)
        name = audio_path.split("/")[-1]
        sample = {"path": name, "sampling_rate": sampling_rate, "array": audio}
        audio_samples.append(sample)
        names.append(name)

    return audio_samples, names


def write_to_file(audio_names: list[str], results: list[dict], audio_in_path: str, audio_out_path: str, save_type: str, return_timestamps: bool = False) -> None:
    """
    Write audio transcription to file.
    :param audio_names: Original audio names
    :param results: Results from ASR
    :param audio_out_path: Path to save audio files
    :param save_type: Type of file to save to (docx, pdf)
    :return: None
    """
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    audio_out_path = os.path.join(audio_out_path, date)
    os.makedirs(audio_out_path, exist_ok=True)

    # For each audio file, create a docx file with the transcription
    for i, (sample, result) in enumerate(zip(audio_names, results)):

        # Copy curr sample from audio in path to audio out path
        import shutil
        shutil.copy(os.path.join(audio_in_path, sample), audio_out_path)
        # Write to file based on the save_type
        for doc_type in save_type.split("_"):
            if doc_type == "docx":
                doc = Document()

                # Set font size and style
                doc.styles['Normal'].font.size = Pt(12)
                doc.styles['Normal'].font.name = 'Arial'

                doc.add_heading(f"Transcription for {sample} - {date}", level=1)
                # Add a line break
                doc.add_paragraph("")
                doc.add_paragraph(result["text"])
                doc.save(os.path.join(audio_out_path, f"transcription_{sample.split('.')[0]}.docx"))
                if return_timestamps:
                    write_timestamps_to_docx(sample, result, date, audio_out_path)
            elif doc_type == "pdf":
                from fpdf import FPDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, f"Transcription for {sample} - {date}", ln=True)
                # Add a line break
                pdf.ln(10)
                # Add the long text using multi_cell
                pdf.multi_cell(0, 10, result["text"])
                pdf.output(os.path.join(audio_out_path, f"transcription_{sample.split('.')[0]}.pdf"))
                if return_timestamps:
                    write_timestamps_to_pdf(sample, result, date, audio_out_path)
            else:
                raise ValueError(f"Invalid save_type {doc_type}")


def write_timestamps_to_docx(sample, result, date, write_path):
    doc = Document()
    # Set font size and style
    doc.styles['Normal'].font.size = Pt(12)
    doc.styles['Normal'].font.name = 'Arial'

    doc.add_heading(f"Transcription with timestamps for {sample} - {date}", level=1)
    # add a line break
    doc.add_paragraph("")
    for chunk in result["chunks"]:
        curr_text = chunk["text"]
        times = chunk["timestamp"]
        start_time = str(times[0]) + " s"
        end_time = str(times[1]) + " s"
        doc.add_paragraph(f"{start_time} - {end_time}: {curr_text}")
    doc.save(os.path.join(write_path, f"transcription_timestamped_{sample.split('.')[0]}.docx"))


def write_timestamps_to_pdf(sample, result, date, write_path):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Transcription with timestamps for {sample} - {date}", ln=True)
    for chunk in result["chunks"]:
        curr_text = chunk["text"]
        times = chunk["timestamp"]
        start_time = str(times[0]) + " s"
        end_time = str(times[1]) + " s"
        pdf.multi_cell(0, 10, f"{start_time} - {end_time}: {curr_text}")
    pdf.output(os.path.join(write_path, f"transcription_timestamped_{sample.split('.')[0]}.pdf"))


def asr(sample: dict, return_timestamps: bool, generate_kwargs: dict = None) -> dict:
    """
    Automatic Speech Recognition (ASR) using Hugging Face's pipeline.
    :param audio_samples: should be a list of dictionaries with keywords "path" and "sampling_rate" and "array"
    :param generate_kwargs:
    :return:
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

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

    result = pipe(sample, return_timestamps=return_timestamps, generate_kwargs=generate_kwargs)
    return result


if __name__ == "__main__":

    # Start timer
    start = datetime.datetime.now()

    print("Loading audio...")
    audio_samples, names = load_audio(audio_in_path)

    print("Running ASR...")
    results = []

    for sample in audio_samples:
        result = asr(sample, return_timestamps=return_timestamps, generate_kwargs=generate_kwargs)
        results.append(result)

    write_to_file(names, results, audio_in_path, audio_out_path, save_type, return_timestamps=return_timestamps)

    # End timer in seconds
    end = datetime.datetime.now()
    print(f"Time taken (s): {end - start}")
    print("Done!")
