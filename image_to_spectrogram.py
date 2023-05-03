import cv2
import numpy as np
import librosa
import soundfile as sf
import os

# sr=22050
filename = 'Arctic_dataset/american_content.wav'
y, sr = librosa.load(filename, sr=16000)
spectrogram = librosa.stft(y)
spectrogram_db = librosa.amplitude_to_db(abs(spectrogram)) # spectrogram_db is in log scale
print(spectrogram_db.shape)

filename = 'Arctic_dataset/american_content_warped.wav'
y, sr = librosa.load(filename, sr=16000)
spectrogram = librosa.stft(y)
spectrogram_db_warped = librosa.amplitude_to_db(abs(spectrogram)) # spectrogram_db is in log scale
print(spectrogram_db_warped.shape)

# cv2.imwrite('american.png', spectrogram_db)
# content_img = cv2.imread('american.png', cv2.IMREAD_GRAYSCALE).astype(np.uint8)
# print(content_img.shape)

directory = 'Results/pretrained'
for root, dirs, files in os.walk(directory):
    for file in files:
        filepath = os.path.join(root, file)
        dtw = root[-12:-4]
        if filepath[-3] == 'wav':
            continue
        # print(os.path.join(root, 'audio.wav'))

        # print(root)
        generated_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if dtw == 'with_dtw':
            generated_img = cv2.resize(generated_img, (104, 1025)).astype(np.uint8)
        else:
            generated_img = cv2.resize(generated_img, (102, 1025)).astype(np.uint8)

        print(generated_img.shape)

        y_inv = librosa.griffinlim(librosa.db_to_amplitude(generated_img)) # reconstruction of audio signal from grayscale image
        sf.write(os.path.join(root, 'audio.wav'), y_inv, sr)