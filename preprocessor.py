import os
import numpy as np
import librosa
import logging
from pathlib import Path
from configparser import ConfigParser


class Preprocessor:
    def __init__(self, config_path="config.ini"):
        self.config = ConfigParser()
        self.config.read(config_path)

        self.sr = self.config.getint("FEATURE_EXTRACTION", "sampling_rate")
        self.output_dim = self.config.getint("FEATURE_EXTRACTION", "output_dim")

        self.train_dir = Path(self.config.get("PATHS", "train_dir"))
        self.val_dir = Path(self.config.get("PATHS", "val_dir"))
        self.test_dir = Path(self.config.get("PATHS", "test_dir"))

    def extract_features(self, file_path):
        y, sr = librosa.load(file_path, sr=self.sr)
        fft = np.abs(np.fft.fft(y))[:self.output_dim]
        stft_128 = np.abs(librosa.stft(y, n_fft=128)).flatten()[:self.output_dim]
        stft_1024 = np.abs(librosa.stft(y, n_fft=1024)).flatten()[:self.output_dim]
        mel_128 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128).flatten()[:self.output_dim]
        mel_1024 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=1024).flatten()[:self.output_dim]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128).flatten()[:self.output_dim]
        return np.concatenate([fft, stft_128, stft_1024, mel_128, mel_1024, mfcc])

    def process_dataset(self, path):
        print('-'*100)
        print(f'Processing: {path}')
        print('-'*100)
        features, labels = [], []
        for label in ['real', 'fake']:
            for ind,file_name in enumerate(os.listdir(path / label)):
                if ind%100 == 0: 

                    print(f'{ind} files processed...')
                if file_name.endswith(".wav"):
                    try:
                        feat = self.extract_features(path / label / file_name)
                        features.append(feat)
                        labels.append(0 if label == 'real' else 1)
                    except Exception as e:
                        logging.warning(f"Failed to process {file_name}: {e}")
                # if ind == 200:
                #     break
        return np.array(features), np.array(labels)

    def prepare_data(self):
        print('-'*100)
        print('Starting preparing Training data.... ')
        print('-'*100)
        X_train, y_train = self.process_dataset(self.train_dir)
        print('-'*100)
        print('Starting preparing Validation data.... ')
        print('-'*100)

        X_val, y_val = self.process_dataset(self.val_dir)
        X_test, y_test = self.process_dataset(self.test_dir)
        return X_train, y_train, X_val, y_val, X_test, y_test
