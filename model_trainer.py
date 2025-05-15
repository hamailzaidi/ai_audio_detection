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
        """
        Constructor method to load training set and path to save the model.
        
        Args:
            X_train (np.ndarray): Feature matrix for training.
            y_train (np.ndarray): True labels for training.
            model_dir (str)     : Path where trained models will be saved.
        """
        
        self.model_dir = Path(model_dir)
        self.X_train = X_train
        self.y_train = y_train
        self.trained_models = []

    def _save_model(self, model, model_name):
        """
        This mehtod dumps the trained model.

        Args:
            model (object): Trained model instance.
            model_name (str): Name of the model.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M")
        self.trained_models.append({'model':model,
                                    'model_name':model_name,
                                    'timestamp':timestamp})
        path = self.model_dir / model_name / timestamp
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path / "model.pkl")
        logging.info(f"Model saved to {path}")

    def train_logistic_regression(self, dump=True):
        """
        This method trains a Logistic Regression model.

        Args:
            dump (bool): Whether to save the model or not.

        Returns:
            LogisticRegression: Trained logistic regression model.
        """

        model = LogisticRegression(max_iter=500, C=1.0, solver='liblinear', random_state=42)
        model.fit(self.X_train, self.y_train)
        if dump:
            self._save_model(model, "LogisticRegression")
        return model

    def train_random_forest(self, n_estimators=100, max_depth=20, dump=True):
        """
        This method trains a Random Forest classifier.

        Args:
            n_estimators (int): Number of trees in the forest.
            max_depth (int)   : Maximum tree depth.
            dump (bool)       : Whether to save the model or not.

        Returns:
            RandomForestClassifier: Trained random forest model.
        """

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(self.X_train, self.y_train)

        if dump:
            self._save_model(model, "RandomForest",)
        return model

    def train_lightgbm(self, dump=True):
        """
        This method trains a LightGBM classifier.

        Args:
            dump (bool): Whether to save the model or not.

        Returns:
            LGBMClassifier: Trained LightGBM model.
        """

        model = lgb.LGBMClassifier(num_leaves=128, max_depth=20, learning_rate=0.05, n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        if dump:
            self._save_model(model, "LightGBM")
        return model
