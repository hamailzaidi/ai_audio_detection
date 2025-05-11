
# Synthetic Audio Detection Project

This projec is a Proof-of-concept for training and evaluating classical machine learning models to detect synthetic audio. It uses frequency based feature extraction like MFCC, FFT, STFTs etc and, for the time being, does not include image (Spectrogram) processing technique. 

The real and artificial audio data used in the training has been taken from open source *Fake or Real* (FoR) dataset which contains around 110K real audios and 80K syntehtic audios. The research paper of FoR has been attached in the **Documentation** folder. The dataset is publically available at https://bil.eecs.yorku.ca/datasets/ 


---

## Project Structure

```
project_root/
├── Documentation             # Research material used to build this project
├── Models/                   # Trained models, and evaluation metrrices
├── config.ini                # Configuration file for paths & parameters
├── preprocessor.py           # Class: Preprocessor: loads and extracts features
├── model_trainer.py          # Class: ModelTrainer: trains ML models
├── model_evaluator.py        # Class: ModelEvaluator: evaluates models
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

All evaluation results are saved to `Models/evaluations/`.

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


## Future work & Limitations

- Potential Feature Engineeering to enhance model performance.
- Hyperparameter tuning
- Test the model on a completely different dataset to analyze its generalizability. 
- Create an inference engine based on the best model.
- Test and Optimize the inference time.
- Create APIs.
- Finalize MLOps Pipeline for future automated retraining and evaluation.


