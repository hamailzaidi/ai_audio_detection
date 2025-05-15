
# Synthetic Audio Detection Project

This projec is a Proof-of-concept for training and evaluating classical machine learning models to detect synthetic audio. It uses frequency based feature extraction like MFCC, FFT, STFTs etc and, for the time being, does not include image (Spectrogram) processing technique. 

The solution implemented is inspired from a 2019 University of Toronto study on differentiating between real and synthetic audios. The data used in the training has been taken from the same project, an open source *Fake or Real* (FoR) dataset which contains around 110K real audios and 80K syntehtic audios. The research paper of FoR has been attached in the **Documentation** folder. The dataset is publically available at https://bil.eecs.yorku.ca/datasets/ 

*FoR dataset* includes synthetic audios from various TTS models (also highlighted in the papers attached), including, 
- DeepVoice3
- Amazon AWS Polly
- Baidu TTS
- Google Traditional TTS
- Google Cloud TTS
- Google Wavenet TTS
- Microsoft Azure TTS

---

## Project Structure

```
project_root/
├── Documentation             # Research material used to build this project
├── Models/                   # Trained models, and evaluation metrrices
├── Datasets/                 # Datasets, contains both for-2sec training data and real-world unseen data to test generalisability
├── config.ini                # Configuration file for paths & parameters
├── preprocessor.py           # Class: Preprocessor: loads and extracts features
├── model_trainer.py          # Class: ModelTrainer: trains ML models
├── model_evaluator.py        # Class: ModelEvaluator: evaluates models
├── predict.py                # Class: InferenceEngine: makes inferences using trained model
├── data_validator.py         # Class: DataValidator: checks for duplication/data leakage across different splits of training data
├── run.py                    # Main file to run the pipeline
├── readme.md                 # Project documentation
├── requirements.txt          # Project dependencies
```

---
## Feature Extraction

The following features are extracted for each audio sample:
- FFT (Fast Fourier Transform)
- STFT (Short-Time Fourier Transform)
- Mel-Spectrogram (128 and 1024 mel bands)
- MFCC (Mel-Frequency Cepstral Coefficients)

Each feature is flattened and concatenated to a fixed-length feature vector.

---

## Models Included

1. Logistic Regression
2. Random Forest
3. LightGBM

---

## Evaluation Metrics

Each model is evaluated on:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

---

## Latest Results
```
Random Forest (50 estimators)

| Accuracy  | Precision | Recall     | F1 Score |
|-----------|-----------|------------|----------|
| 0.90625   | 0.84639   | 0.99265    | 0.91371  |

LightGBM (128 num_leaves)

| Accuracy  | Precision | Recall     | F1 Score |
|-----------|-----------|------------|----------|
| 0.93658   | 0.93577   | 0.9375     | 0.93663  |

Logistic Regression

| Accuracy  | Precision | Recall     | F1 Score |
|-----------|-----------|------------|----------|
| 0.74448   | 0.81666   | 0.63051    | 0.71161  |

```

## Addressing Model Generalizability

While the model achieves high accuracy on some test sets, its performance drops significantly on others. This highlights a core challenge in deepfake audio detection: **generalizing across unseen TTS systems and varying synthetic voice characteristics**.

### Inference Results Across Datasets

| Dataset Name                  | True Label | Total Samples | Classified as REAL | Classified as FAKE | Accuracy |
|------------------------------|------------|----------------|---------------------|---------------------|----------|
| LJspeech                     | real       | 200            | 191                 | 9                   | 0.95     |
| demo_tts_dataset_hf          | fake       | 5              | 4                   | 1                   | 0.20     |
| tts_audio_samples_hf         | fake       | 20             | 0                   | 20                  | 1.00     |
| tts_audio_samples_hf2        | fake       | 20             | 2                   | 18                  | 0.90     |
| tts_english_may_10_hf        | fake       | 274            | 217                 | 57                  | 0.21     |

These results show the model performs exceptionally on certain TTS datasets (e.g., 100% on `tts_audio_samples_hf`) but poorly on others (e.g., ~20% on `demo_tts_dataset_hf` and `tts_english_may_10_hf`).

### Why This might be happening

- The model may be overfitting to specific TTS characteristics in the training data.
- TTS systems evolve rapidly — newer models differ in audio quality, cadence, prosody, and noise profiles.
- Lack of diverse negative (synthetic) examples during training.

## Strategies to Improve Generalizability

To improve the model generalisability against a wider range of synthetic voices and TTS technologies, future effort of this project will incorporate additional public datasets. These datasets provide diverse sources of real and synthetic speech which will enable better learning of audio representations.


### Potential New Datasets:

- **ASVspoof 2021 Challenge - Speech Deepfake Database**  
  [https://zenodo.org/records/4835108](https://zenodo.org/records/4835108)  
  A benchmark dataset for spoofing countermeasures in automatic speaker verification systems.

- **WaveFake: A Data Set to Facilitate Audio Deepfake Detection**  
  [https://zenodo.org/records/5642694](https://zenodo.org/records/5642694)  
  A comprehensive deepfake dataset including speech synthesized using modern neural vocoders.

- **LJSpeech Dataset**  
  [https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/)  
  A widely used dataset for TTS training. Real, high-quality speech from a single female speaker.

- **LibriTTS Corpus**  
  [https://openslr.org/60/](https://openslr.org/60/)  
  A multi-speaker corpus for TTS research, based on public domain LibriSpeech audiobook recordings.

These datasets can be used to:
- Retrain existing models with more variety
- Fine-tune on recent synthetic voices
- Benchmark the system's performance on new, unseen synthetic speech types

