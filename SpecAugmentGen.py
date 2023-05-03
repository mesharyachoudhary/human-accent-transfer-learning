import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os
import math
import random
from scipy.io import wavfile
import random
random.seed(111)

#directory where dataset to be augmented is located
HOME_DIR = "/home/ubuntu/Downloads"
#name for the augmented dataset
NEWDATASET = "SpecAugmentDataset"
DATASET_DIR = HOME_DIR+"/"+"Dataset"
# creates the training and validation directory in the augmented dataset
os.makedirs("/home/ubuntu/Downloads/SpecAugmentDataset/training", exist_ok=True)
os.makedirs("/home/ubuntu/Downloads/SpecAugmentDataset/validation", exist_ok=True)

# SpecAugment function which performs time,frquency masking over spectrograms for data augmentation
def spec_augment(spec: np.ndarray, num_mask=2, 
                 freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):

    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
        
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)
        
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0
    
    return spec

for dirpath, dirnames, filenames in os.walk(DATASET_DIR):
    # iterate over all the directories(corresponding to various accents) in the arctic dataset
    for dirname in dirnames:
        CUR_DIR = DATASET_DIR+"/"+dirname
        accent = dirname
        # iterate over all the files corresponding to an accent
        for dirpath, dirnames, filenames in os.walk(CUR_DIR):
            i=1
            N = len(filenames)
            # we want the validation set to contain 10% of the samples
            M = math.floor(N/10)
            for filename in filenames:
                # randomly choose M files from the total N files
                random_number = random.randint(0,N)
                datasettype = ""
                if random_number<=M:
                       datasettype = "validation"
                       N-=1
                       M-=1
                else:
                       datasettype = "training"
                       N-=1
                filelocation = dirpath+"/"+filename
                y, sr = librosa.load(filelocation)
                # generate the melspectrogram
                mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
                mel_spect_amp_db = librosa.power_to_db(mel_spect)
                # perform spec augmentation of the spectrogram
                spec_aug_mel = spec_augment(mel_spect_amp_db)
                # new directory for the augmented dataset
                newfolder = HOME_DIR+"/"+NEWDATASET+"/"+datasettype+"/"+accent
                os.makedirs(newfolder, exist_ok=True)
                outputfilename = HOME_DIR+"/"+NEWDATASET+"/"+datasettype+"/"+accent+"/"+str(i)+".png"
                # saves the augmented spectrogram
                plt.imsave(outputfilename, spec_aug_mel, cmap='gray' )
                i+=1

