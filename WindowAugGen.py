import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os
import math
import random

#directory where dataset to be augmented is located
HOME_DIR = "/home/ubuntu/Downloads"
#name for the augmented dataset
NEWDATASET = "NewDataset"
DATASET_DIR = HOME_DIR+"/"+"Dataset"
# creates the training and validation directory in the augmented dataset
os.makedirs("/home/ubuntu/Downloads/NewDataset/testing", exist_ok=True)
os.makedirs("/home/ubuntu/Downloads/NewDataset/validation", exist_ok=True)
for dirpath, dirnames, filenames in os.walk(DATASET_DIR):
    # iterate over all the directories(corresponding to various accents) in the arctic dataset
    for dirname in dirnames:
        CUR_DIR = DATASET_DIR+"/"+dirname
        accent = dirname
        # iterate over all the files corresponding to an accent
        for dirpath, dirnames, filenames in os.walk(CUR_DIR):
            i=1
            N = len(filenames)
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
                # creates spectrograms with nfft = 256, 512, 1024, 2048
                for k in range(0,4):
                    nfft = 256*(2**(k))
                    # generates spectrogram
                    y_style, sr_style = librosa.load(filelocation, sr=None)
                    spectrogram_style = librosa.stft(y_style, n_fft=nfft)
                    spectrogram_db_style = librosa.amplitude_to_db(abs(spectrogram_style)) # spectrogram_db is in log scale
                    # new directory for the augmented dataset
                    newfolder = HOME_DIR+"/"+NEWDATASET+"/"+datasettype+"/"+accent
                    os.makedirs(newfolder, exist_ok=True)
                    outputfilename = HOME_DIR+"/"+NEWDATASET+"/"+datasettype+"/"+accent+"/"+str(i)+"_"+str(nfft)+".png"
                    # saves the augmented spectrogram
                    plt.imsave(outputfilename, spectrogram_db_style, cmap='gray') # saving spectrogram_db as a png file                       
                i+=1