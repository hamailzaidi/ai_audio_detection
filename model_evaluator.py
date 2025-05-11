import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class ModelEvaluator:
    def __init__(self, output_dir="Models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, model, X, y):
        y_pred = model.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred),
            'conf_matrix': confusion_matrix(y, y_pred).tolist()
        }

    def save_evaluation(self, model, val_metrics, test_metrics):
        eval_path = self.output_dir / model.get('model_name') / model.get('timestamp')
        eval_path.mkdir(parents=True, exist_ok=True)

        pd.Series(val_metrics).to_csv(eval_path / "validation_metrics.csv")
        pd.Series(test_metrics).to_csv(eval_path / "test_metrics.csv")
        logging.info(f"Metrics saved for {model.get('model_name')} to {eval_path}")
