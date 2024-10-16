# Use ffmpeg that converts a video file to an audio file using ffmpeg
# The audio file is saved in the same directory as the video file. Audio is located in ../in

import os
import subprocess



if __name__ == "__main__":
    file_name = 'Begroting-Water-Plenaire-zaal-2024-10-08.mp4'
    file_path = os.path.join('..', 'in', file_name)
    audio_path = os.path.join('..', 'in', file_name.replace('.mp4', '.mp3'))
    # Convert video to audio
    subprocess.run(['ffmpeg', '-i', file_path, audio_path])
    print(f'Audio file saved at {audio_path}')


