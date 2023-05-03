import librosa
import os
import numpy as np
import cv2
import soundfile as sf
import re

# Define the input file path
input_file_path = "./archive/recordings/recordings/chittagonian1.mp3"

# Define the output directory
output_dir = "./clips/"

# Define the duration of each clip in seconds
clip_duration = 5

# Load the audio signal from the input file
y, sr = librosa.load(input_file_path)
print(y.shape)
print(sr)

n_fft = 2048
hop_length = n_fft // 4

# Calculate the total number of samples for each clip
clip_samples = int(clip_duration * sr)

# Calculate the total number of clips that can be extracted from the audio signal
num_clips = int(np.floor(len(y) / clip_samples))

# Extract the clips and save them as WAV files
for i in range(num_clips):
    # Calculate the start and end sample indices for the current clip
    start_sample = i * clip_samples
    end_sample = (i + 1) * clip_samples

    # Extract the current clip from the audio signal
    clip = y[start_sample:end_sample]
    clip = librosa.util.normalize(clip)

    spectrogram = librosa.stft(clip, n_fft=n_fft, hop_length=hop_length)
    spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
    print(spectrogram_db.shape)


    # Define the output file path for the current clip
    output_file_path = os.path.join(output_dir, f"clip_{i}")

    # Save the current clip as a WAV file
    cv2.imwrite(f'{output_file_path}.png', spectrogram_db)
    img = cv2.imread(f'{output_file_path}.png', cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    y_inv = librosa.griffinlim(librosa.db_to_amplitude(img))
    sf.write(f'{output_file_path}.wav', y_inv, sr)


# mp3_dir = './archive/recordings/recordings'
# clip_duration = 5
# n_fft = 2048
# hop_length = n_fft // 4
# output_dir = "./clips/"

# for filename in os.listdir(mp3_dir):
#     if filename.endswith('.mp3'):
#         # Load the audio file
#         filepath = os.path.join(mp3_dir, filename)
#         y, sr = librosa.load(filepath)
#         print(sr)
#         print(y.shape)
        
#         # Get the prefix of the filename
#         prefix = filename.replace(".mp3", "")
#         print(prefix)
#         parts = re.split('(\d+)', prefix, maxsplit=1)
#         # print(parts[1])
#         clips_dir = os.path.join(output_dir, parts[0])
#         os.makedirs(clips_dir, exist_ok=True)

#         # Calculate the total number of samples for each clip
#         clip_samples = int(clip_duration * sr)

#         # Calculate the total number of clips that can be extracted from the audio signal
#         num_clips = int(np.floor(len(y) / clip_samples))

#         # Extract the clips and save them as WAV files
#         for i in range(num_clips):
#             # Calculate the start and end sample indices for the current clip
#             start_sample = i * clip_samples
#             end_sample = (i + 1) * clip_samples

#             # Extract the current clip from the audio signal
#             clip = y[start_sample:end_sample]
#             clip = librosa.util.normalize(clip)

#             spectrogram = librosa.stft(clip, n_fft=n_fft, hop_length=hop_length)
#             spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
#             print(spectrogram_db.shape)

#             # Define the output file path for the current clip
#             output_file_path = os.path.join(clips_dir, f"clip_{parts[1]}_{i}")

#             # Save the current clip as a WAV file
#             cv2.imwrite(f'{output_file_path}.png', spectrogram_db)

