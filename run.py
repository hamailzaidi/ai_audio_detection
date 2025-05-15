import logging
from preprocessor import Preprocessor,Preprocessor2
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO)

# Load and process data
pre = Preprocessor()
X_train, y_train, X_val, y_val, X_test, y_test = pre.prepare_data()

# Train models
trainer = ModelTrainer(X_train,y_train)
models = {
    "LogisticRegression": trainer.train_logistic_regression(),
    "RandomForest": trainer.train_random_forest(n_estimators=50,max_depth=20),
    "LightGBM": trainer.train_lightgbm()
}

# Evaluate models
evaluator = ModelEvaluator()
for model in trainer.trained_models:
    val_metrics = evaluator.evaluate(model.get('model'), X_val, y_val)
    test_metrics = evaluator.evaluate(model.get('model'), X_test, y_test)
    evaluator.save_evaluation(model, val_metrics, test_metrics)
