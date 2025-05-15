import joblib
import logging
import os
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

class ModelTrainer:
    def __init__(self, X_train, y_train, model_dir="Models"):
        self.model_dir = Path(model_dir)
        self.X_train = X_train
        self.y_train = y_train
        self.trained_models = []

    def _save_model(self, model, model_name):

        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M")
        self.trained_models.append({'model':model,
                                    'model_name':model_name,
                                    'timestamp':timestamp})
        path = self.model_dir / model_name / timestamp
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path / "model.pkl")
        logging.info(f"Model saved to {path}")

    def train_logistic_regression(self, dump=True):
        model = LogisticRegression(max_iter=500, C=1.0, solver='liblinear', random_state=42)
        model.fit(self.X_train, self.y_train)
        if dump:
            self._save_model(model, "LogisticRegression")
        return model

    def train_random_forest(self, n_estimators=100, max_depth=20, dump=True):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(self.X_train, self.y_train)


        if dump:
            self._save_model(model, "RandomForest",)
        return model

    def train_lightgbm(self, dump=True):
        model = lgb.LGBMClassifier(num_leaves=128, max_depth=20, learning_rate=0.05, n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        if dump:
            self._save_model(model, "LightGBM")
        return model
