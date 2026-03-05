import os
import sys
import numpy as np
import joblib
from dataclasses import dataclass

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    model_path: str = "artifacts/model_trainer/model.pkl"
    threshold_path: str = "artifacts/model_trainer/threshold.pkl"


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    # --------------------------------------------------------
    # Dynamically find best threshold using F1 Score
    # --------------------------------------------------------
    def find_best_threshold(self, y_true, y_probs):
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

        # Avoid division by zero
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        best_index = np.argmax(f1_scores)

        # thresholds array is shorter than precision/recall by 1
        if best_index >= len(thresholds):
            return 0.5  # safe default

        return thresholds[best_index]

    # --------------------------------------------------------
    # Model Training Pipeline
    # --------------------------------------------------------
    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Starting SMOTE for class balancing")

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            logging.info("SMOTE completed")

            # Auto scale_pos_weight based on imbalance
            negative = np.sum(y_train == 0)
            positive = np.sum(y_train == 1)
            scale_pos_weight = negative / (positive + 1e-8)

            logging.info(f"Scale Pos Weight: {scale_pos_weight}")

            # Strong XGBoost configuration
            model = XGBClassifier(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=7,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=1,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
                use_label_encoder=False
            )

            logging.info("Training XGBoost model")

            model.fit(X_resampled, y_resampled)

            # ----------------------------
            # Evaluation
            # ----------------------------
            y_probs = model.predict_proba(X_test)[:, 1]

            roc_score = roc_auc_score(y_test, y_probs)
            pr_auc = average_precision_score(y_test, y_probs)

            logging.info(f"ROC AUC Score: {roc_score}")
            logging.info(f"PR AUC Score: {pr_auc}")

            print("ROC AUC:", roc_score)
            print("PR AUC:", pr_auc)

            # ----------------------------
            # Threshold Optimization
            # ----------------------------
            best_threshold = min(self.find_best_threshold(y_test, y_probs), 0.35)

            logging.info(f"Optimal Threshold: {best_threshold}")
            print("Optimal Threshold:", best_threshold)

            # ----------------------------
            # Save model & threshold
            # ----------------------------
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)

            joblib.dump(model, self.config.model_path)
            joblib.dump(best_threshold, self.config.threshold_path)

            logging.info("Model and threshold saved successfully")

            return roc_score

        except Exception as e:
            raise CustomException(e, sys)