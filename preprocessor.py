import os
import numpy as np
import librosa
import logging
from pathlib import Path
from configparser import ConfigParser
from sklearn.model_selection import train_test_split

class Preprocessor:

    """
    A class for loading audio data and extracting features for training, validation, and testing datasets.
    Feature extraction supports FFT, STFT, MFCC, Mel-spectrogram etc..

    """

    def __init__(self, config_path="config.ini"):
        """
        Initializes the Preprocessor using parameters from a configuration file.

        Args:
            config_path (str): Path to the configuration file.
        """

        self.config = ConfigParser()
        self.config.read(config_path)

        self.sr = self.config.getint("FEATURE_EXTRACTION", "sampling_rate")
        self.output_dim = self.config.getint("FEATURE_EXTRACTION", "output_dim")

        self.train_dir = Path(self.config.get("PATHS", "train_dir"))
        self.val_dir = Path(self.config.get("PATHS", "val_dir"))
        self.test_dir = Path(self.config.get("PATHS", "test_dir"))

    # ------------------------------------------------------------------------------------ 
    # Different  Feature Engineering techniques employed and tested against the evaluation metrics.


    def _extract_features(self, y):
        """
        Extracts a fixed-length feature vector from an audio signal using multiple spectral techniques.
        This implementation is same as of Fake or Real (FoR) dataset paper.

        Args:
            y (np.ndarray): Audio signal.

        Returns:
            np.ndarray: Feature vector.
        """
        
        fft = np.abs(np.fft.fft(y))[:self.output_dim]
        stft_128 = np.abs(librosa.stft(y, n_fft=128)).flatten()[:self.output_dim]
        stft_1024 = np.abs(librosa.stft(y, n_fft=1024)).flatten()[:self.output_dim]
        mel_128 = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128).flatten()[:self.output_dim]
        mel_1024 = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=1024).flatten()[:self.output_dim]
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=128).flatten()[:self.output_dim]
        return np.concatenate([fft, stft_128, stft_1024, mel_128, mel_1024, mfcc])

    def _extract_features_v2(self,y, n_mfcc=30):

        """
        Extracts MFCC-based features including deltas, zero crossing rate, and spectral flatness.

        Args:
            y (np.ndarray): Audio signal.
            n_mfcc (int): Number of MFCCs to compute.

        Returns:
            np.ndarray: Feature vector.
        """

        # print(f"Extracting features from: {file_path}")

        # MFCCs and deltas
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Flatten temporal dimension using mean and std
        mfccs = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(delta, axis=1), np.std(delta, axis=1),
            np.mean(delta2, axis=1), np.std(delta2, axis=1)
        ])

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        flat_mean = np.mean(flatness)
        flat_std = np.std(flatness)

        # This has been commented out because computing fundamental frequency takes much more time around 0.6 seconds per file
        # as compared to the other features. 
        
        # Fundamental frequency (pitch)
        # f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=self.sr)
        # f0 = f0[~np.isnan(f0)]  # Remove NaNs
        # if len(f0) > 0:
        #     pitch_mean = np.mean(f0)
        #     pitch_std = np.std(f0)
        # else:
        #     pitch_mean, pitch_std = 0, 0

        # Final feature vector
        features = np.concatenate([
            mfccs,
            [zcr_mean, zcr_std],
            [flat_mean, flat_std],
            # [pitch_mean, pitch_std]
        ])

        # print(features)
        return features

    def _extract_features_v3(self, y, n_mfcc=30):
        """
        Extended feature extraction including MFCC, spectral stats, FFT, STFT, and spectral contrast.

        Args:
            y (np.ndarray): Audio signal.
            n_mfcc (int): Number of MFCCs.

        Returns:
            np.ndarray: Feature vector.
        """

        # MFCCs and their deltas
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        mfccs = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(delta, axis=1), np.std(delta, axis=1),
            np.mean(delta2, axis=1), np.std(delta2, axis=1)
        ])

        # ================================
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        # ================================
        # Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        flat_mean = np.mean(flatness)
        flat_std = np.std(flatness)

        # ================================
        # FFT Features (Magnitude Spectrum)
        fft_spectrum = np.fft.fft(y)
        magnitude = np.abs(fft_spectrum)
        fft_mean = np.mean(magnitude)
        fft_std = np.std(magnitude)

        # ================================
        # STFT Features
        stft = np.abs(librosa.stft(y))
        centroid = librosa.feature.spectral_centroid(S=stft, sr=self.sr)
        bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=self.sr)
        rolloff = librosa.feature.spectral_rolloff(S=stft, sr=self.sr)
        contrast = librosa.feature.spectral_contrast(S=stft, sr=self.sr)

        stft_features = np.concatenate([
            np.mean(centroid, axis=1), np.std(centroid, axis=1),
            np.mean(bandwidth, axis=1), np.std(bandwidth, axis=1),
            np.mean(rolloff, axis=1), np.std(rolloff, axis=1),
            np.mean(contrast, axis=1), np.std(contrast, axis=1)
        ])

        # ================================
        # Final feature vector
        features = np.concatenate([
            mfccs,
            [zcr_mean, zcr_std],
            [flat_mean, flat_std],
            [fft_mean, fft_std],
            stft_features
        ])

        return features


    def _extract_spectral_features(self, y):
        """
        Extracts the average spectral centroid, bandwidth, and rolloff from an audio signal.

        Args:
            y (np.ndarray): Audio signal.

        Returns:
            np.ndarray: Feature vector with mean values of spectral features.
        """

        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=self.sr))

        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=self.sr))

        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=self.sr))

        features = np.array([centroid, bandwidth, rolloff])

        return features




    # ------------------------------------------------------------------------------------

    # Getter method for the selected feature engineering technique
    def engineer_features(self,y):
        """
        Feature engineering wrapper to select the selected feature engineering method.

        Args:
            y (np.ndarray): Audio signal.

        Returns:
            np.ndarray: Feature vector.
        """

        # return self._extract_features(y) 
        return self._extract_spectral_features(y)


    def _process_dataset(self, path):
        """
        Loads and processes audio files from a specific dataset directory.

        Args:
            path (Path): Path to dataset directory with 'real' and 'fake' subfolders.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels arrays.
        """

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
                        file_path = path / label / file_name
                        y, sr = librosa.load(path=file_path, sr=self.sr)
                        feat = self.engineer_features(y)
                        # break
                        features.append(feat)
                        labels.append(0 if label == 'real' else 1)
                    except Exception as e:
                        logging.warning(f"Failed to process {file_name}: {e}")
                # if ind == 200:
                #     break
        return np.array(features), np.array(labels)

    def prepare_data(self):
        """
        Loads and processes training, validation, and test data from configured paths.

        Returns:
            Tuple of np.ndarrays: X_train, y_train, X_val, y_val, X_test, y_test
        """

        print('-'*100)
        print('Starting preparing Training data.... ')
        print('-'*100)
        X_train, y_train = self._process_dataset(self.train_dir)
        print('-'*100)
        print('Starting preparing Validation data.... ')
        print('-'*100)

        X_val, y_val = self._process_dataset(self.val_dir)
        X_test, y_test = self._process_dataset(self.test_dir)
        return X_train, y_train, X_val, y_val, X_test, y_test



class Preprocessor2:
    """
    Alternate preprocessor for handling a single dataset split and applying feature extraction + splitting.
    """

    def __init__(self, config_path="config.ini"):
        """
        Initializes with a config file pointing to real and fake audio folders.

        Args:
            config_path (str): Path to the config file.
        """

        self.config = ConfigParser()
        self.config.read(config_path)

        self.sr = self.config.getint("FEATURE_EXTRACTION", "sampling_rate")
        self.output_dim = self.config.getint("FEATURE_EXTRACTION", "output_dim")

        self.real_dir = Path(self.config.get("PATHS", "real_dir"))
        self.fake_dir = Path(self.config.get("PATHS", "fake_dir"))

    def extract_features_v2(self, file_path, n_mfcc=30):
        """
        This method extracts MFCC + delta, ZCR, and flatness features from a single audio file.

        Args:
            file_path (str): Path to .wav file.
            n_mfcc (int)   : Number of MFCCs.

        Returns:
            np.ndarray: Feature vector.
        """


        y, sr = librosa.load(file_path, sr=self.sr)

        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        mfccs = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(delta, axis=1), np.std(delta, axis=1),
            np.mean(delta2, axis=1), np.std(delta2, axis=1)
        ])

        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        flatness = librosa.feature.spectral_flatness(y=y)
        flat_mean = np.mean(flatness)
        flat_std = np.std(flatness)

        features = np.concatenate([
            mfccs,
            [zcr_mean, zcr_std],
            [flat_mean, flat_std]
        ])
        return features

    def process_dataset(self, real_path, fake_path):
        """
        Loads and processes all .wav files from real and fake directories.

        Args:
            real_path (Path): Directory of real samples.
            fake_path (Path): Directory of fake samples.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Feature and label arrays.
        """


        print('-'*100)
        print(f'Processing Real and Fake data...')
        print('-'*100)
        features, labels = [], []

        for label_dir, label_value in zip([real_path, fake_path], [0, 1]):
            label_name = 'real' if label_value == 0 else 'fake'
            print(f"Processing {label_name} files in: {label_dir}")
            for idx, file_name in enumerate(os.listdir(label_dir)):
                if idx % 100 == 0:
                    print(f'{idx} {label_name} files processed...')
                if file_name.endswith(".wav"):
                    try:
                        file_path = label_dir / file_name
                        feat = self.extract_features_v2(file_path)
                        features.append(feat)
                        labels.append(label_value)
                    except Exception as e:
                        logging.warning(f"Failed to process {file_name}: {e}")
        return np.array(features), np.array(labels)

    def prepare_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """
        Loads features and performs train/val/test split with stratification.

        Args:
            test_size (float): Fraction of data to reserve for test.
            val_size (float): Fraction to reserve for validation.
            random_state (int): Random seed.

        Returns:
            Tuple of np.ndarrays: X_train, y_train, X_val, y_val, X_test, y_test
        """


        print('-'*100)
        print('Extracting features and splitting dataset...')
        print('-'*100)
        X, y = self.process_dataset(self.real_dir, self.fake_dir)

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        val_ratio_adjusted = val_size / (1 - test_size)  # Adjust val size relative to train+val
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_ratio_adjusted,
            stratify=y_trainval, random_state=random_state
        )

        print("✅ Data splitting complete.")
        print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        return X_train, y_train, X_val, y_val, X_test, y_test
