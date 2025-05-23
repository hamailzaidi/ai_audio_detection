import numpy as np
import librosa
import joblib
import logging
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessor import Preprocessor
from datetime import datetime 
import warnings
from configparser import ConfigParser
warnings.filterwarnings("ignore")


class InferenceEngine:
    """
    A class that does audio classification inference using a pre-trained model.
    """


    def __init__(self, preprocessor: Preprocessor, chunk_duration=2.0,config_path = 'config.ini'):
        """
        Constructor method that loads the model and configurations.

        Args:
            preprocessor (Preprocessor): An instance of the Preprocessor class.
            chunk_duration (float): Duration (in seconds) to split audio into chunks.
            config_path (str): Path to the configuration file with model path.
        """

        self.config = ConfigParser()
        self.config.read(config_path)

        self.model_path = self.config.get("PATHS", "selected_model_path")
        self.model = joblib.load(self.model_path)
        self.preprocessor = preprocessor
        self.chunk_samples = int(chunk_duration * preprocessor.sr)
        self.output_dim = 1024

    

    def _split_audio(self, y):
        """
        This method splits audio into chunks of predefined length.
        (2 sec in our case, because the training was done on a dataset of 2 sec audios.)

        Args:
            y (np.ndarray): Audio Signal.

        Returns:
            List[np.ndarray]: List of audio chunks.
        """

        return [y[start:start + self.chunk_samples]
                for start in range(0, len(y) - self.chunk_samples + 1, self.chunk_samples)]
      

    def predict_file(self, file_path,split=True):
        """
        This method predicts the label of a given audio file by averaging predictions across chunks.

        Args:
            file_path (str): Path to the .wav file.

        Returns:
            int: 0 (REAL) or 1 (FAKE) or None if the file is too short.
        """


        y, sr = librosa.load(file_path, sr=self.preprocessor.sr)
        if len(y) < self.chunk_samples:
            y = np.pad(y, (0, self.chunk_samples - len(y)))
        if split: 
            chunks = self._split_audio(y)
            if not chunks:
                return None

            features = [self.preprocessor.engineer_features(chunk) for chunk in chunks]
            preds = self.model.predict(np.array(features))
            mean_pred = np.mean(preds)
            return 1 if mean_pred == 0.5 else int(np.round(mean_pred))
        else: 
            features = self.preprocessor.engineer_features(y)
            scaler = joblib.load('TestPipeline\scaler.joblib')
            features_scaled = scaler.transform([features])
            pred = self.model.predict(features_scaled)[0]

            # in the trained model, 0 means AI, 1 means Human, and in this pipeline, 1 means AI so
            # thats why.
            return 1 if pred<=0.5 else 0
            pass

    def run_on_directory(self, root_dir, output_base="InferenceResults"):
        """
        This method runs inference on a test dataset directory structured with 'real/' and 'fake/' subfolders.

        Args:
            root_dir (str): Root directory containing labeled subfolders.
            output_base (str): Base directory where results will be saved.
        """


        root = Path(root_dir)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(output_base) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        log_path = output_dir / "inference.log"
        self.logger = self._setup_logger(log_path)

        results_summary = []

        for label_folder in ['real', 'fake']:
            true_label = 0 if label_folder == 'real' else 1
            label_path = root / label_folder

            for dataset_path in sorted(label_path.iterdir()):
                if not dataset_path.is_dir():
                    continue

                self.logger.info("-" * 100)
                self.logger.info(f"Model: {self.model}")
                self.logger.info(f"Model Path: {self.model_path}")
                self.logger.info(f"Dataset: {dataset_path}")
                self.logger.info(f"True Label: {label_folder}")

                preds, filenames = [], []

                for wav_file in sorted(dataset_path.glob("*.wav")):
                    pred = self.predict_file(wav_file,split=False)
                    if pred is not None:
                        preds.append(pred)
                        filenames.append(wav_file.name)

                if not preds:
                    self.logger.info(" No valid files found.")
                    continue

                y_true = [true_label] * len(preds)
                acc = accuracy_score(y_true, preds)
                prec = precision_score(y_true, preds)
                rec = recall_score(y_true, preds)
                f1 = f1_score(y_true, preds)

                self.logger.info(f"Total files processed: {len(preds)}")
                self.logger.info(f"Classified as REAL (0): {preds.count(0)}")
                self.logger.info(f"Classified as FAKE (1): {preds.count(1)}")
                self.logger.info(f"Accuracy: {acc:.2f}\n")

                result_df = pd.DataFrame({
                    "file": filenames,
                    "true_label": y_true,
                    "predicted": preds
                })
                csv_name = f"{dataset_path.name}_results.csv"
                result_df.to_csv(output_dir / csv_name, index=False)

                results_summary.append({
                    "dataset": dataset_path.name,
                    "true_label": label_folder,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1
                })

        pd.DataFrame(results_summary).to_csv(output_dir / "evaluation_summary.csv", index=False)
        self.logger.info("✅ All evaluations completed and saved.")


    def predict(self, file_path):
        """
        This method predicts and prints the label of a single audio file. used for separate inferencing of a single file, 
        it uses the predict_file method interenally. 

        Args:
            file_path (str): Path to the .wav file.

        Returns:
            Tuple[str, int] or None: (Label, Prediction) or None if invalid.
        """
        
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return None

        prediction = self.predict_file(file_path,split=False)
        if prediction is not None:
            label = "Fake" if prediction else "Real"
            print(f"{file_path.name} Predicted: {label} ({prediction})")
            return label, prediction
        else:
            print(f"{file_path.name} Skipped (invalid or too short)")
            return None


    @staticmethod
    def _setup_logger(path):
        """
        Initializes a logger that writes to a specified file.

        Args:
            path (Path): Path to log file.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger("inference_logger")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(fh)
        return logger


if __name__ == "__main__":
    preprocessor = Preprocessor("config.ini")
    engine = InferenceEngine(
        preprocessor=preprocessor,

    )

    # Run full evaluation
    engine.run_on_directory(root_dir="Datasets/unseen_data")

    # Predict single file
    # engine.predict("Datasets/unseen_data/fake/tts_audio_samples_hf2/output_0007.wav")
