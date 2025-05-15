import os
import hashlib
from pathlib import Path
import librosa
import numpy as np
from configparser import ConfigParser


class DataValidator:

    def __init__(self, config_path="config.ini"):
        self.config = ConfigParser()
        self.config.read(config_path)

        self.real_dir = Path(self.config.get("PATHS", "real_dir"))
        self.fake_dir = Path(self.config.get("PATHS", "fake_dir"))

    def hash_audio(self, file_path, sr=16000):
        """Load audio and compute a hash based on its waveform."""
        try:
            y, _ = librosa.load(file_path, sr=sr, mono=True)
            print(type(y))
            y = y / np.max(np.abs(y))  # Normalize
            return hashlib.sha256(y.tobytes()).hexdigest()
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None

    def validate_dataset(self, sr=16000):
        print("\n🔍 Validating dataset for duplicate or overlapping audio and filenames...")
        real_dir = self.real_dir
        fake_dir = self.fake_dir

        real_hashes = {}
        fake_hashes = {}
        real_names = {}
        fake_names = {}

        total_real = 0
        total_fake = 0

        print("\n📂 Processing REAL audio files...")
        for i, file in enumerate(real_dir.glob("*.wav")):
            if i % 100 == 0:
                print(f'{i} files processed')
            h = self.hash_audio(file, sr=sr)
            
            name = file.name

            if name in real_names:
                print(f"⚠️ DUPLICATE FILENAME in REAL: {name} == {real_names[name].name}")
            else:
                real_names[name] = file

            if h:
                if h in real_hashes:
                    print(f"⚠️ DUPLICATE AUDIO in REAL: {file.name} == {real_hashes[h].name}")
                else:
                    real_hashes[h] = file

            total_real += 1

        print("\n📂 Processing FAKE audio files...")
        for i, file in enumerate(fake_dir.glob("*.wav")):
            if i % 100 == 0:
                print(f'{i} files processed')
            h = self.hash_audio(file, sr=sr)
            name = file.name

            if name in fake_names:
                print(f"⚠️ DUPLICATE FILENAME in FAKE: {name} == {fake_names[name].name}")
            else:
                fake_names[name] = file

            if h:
                if h in fake_hashes:
                    print(f"⚠️ DUPLICATE AUDIO in FAKE: {file.name} == {fake_hashes[h].name}")
                else:
                    fake_hashes[h] = file

            total_fake += 1

        # Cross-set validation
        print("\n🔁 Checking for overlaps between REAL and FAKE...")

        overlapping_hashes = set(real_hashes.keys()) & set(fake_hashes.keys())
        for h in overlapping_hashes:
            print(f"❌ SAME AUDIO in both REAL and FAKE: {real_hashes[h].name} == {fake_hashes[h].name}")

        overlapping_names = set(real_names.keys()) & set(fake_names.keys())
        for name in overlapping_names:
            print(f"❌ SAME FILENAME in both REAL and FAKE: {name}")

        print("\n✅ Validation complete.")
        print("📊 Summary:")
        print(f"Total real files checked: {total_real}")
        print(f"Total fake files checked: {total_fake}")
        print(f"Unique audio hashes in REAL: {len(real_hashes)}")
        print(f"Unique audio hashes in FAKE: {len(fake_hashes)}")
        print(f"Duplicate audio files in REAL: {total_real - len(real_hashes)}")
        print(f"Duplicate audio files in FAKE: {total_fake - len(fake_hashes)}")
        print(f"Overlapping audio (REAL ↔ FAKE): {len(overlapping_hashes)}")
        print(f"Overlapping filenames (REAL ↔ FAKE): {len(overlapping_names)}")




if __name__ == "__main__":

    DataValidator().validate_dataset()