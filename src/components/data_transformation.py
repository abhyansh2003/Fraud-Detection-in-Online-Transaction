import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from dataclasses import dataclass
import joblib

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = "artifacts/data_transformation/preprocessor.pkl"


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Feature Engineering

            train_df["amount_ratio"] = train_df["amount"] / (train_df["avg_amount_last_24h"] + 1)
            test_df["amount_ratio"] = test_df["amount"] / (test_df["avg_amount_last_24h"] + 1)

            train_df["high_amount_flag"] = (train_df["amount"] > 10000).astype(int)
            test_df["high_amount_flag"] = (test_df["amount"] > 10000).astype(int)

            train_df["night_transaction"] = (train_df["hour_of_day"] <= 5).astype(int)
            test_df["night_transaction"] = (test_df["hour_of_day"] <= 5).astype(int)
            
            selected_features = [
                "amount",
                "txn_count_last_24h",
                "avg_amount_last_24h",
                "device_trust_score",
                "ip_address_risk_score",
                "merchant_historical_fraud_rate",
                "otp_success_rate_customer",
                "past_fraud_count_customer",
                "location_change_flag",
                "device_change_flag",
                "is_international",
                "hour_of_day",
                "is_weekend",
                # new features
                "amount_ratio",
                "high_amount_flag",
                "night_transaction"
            ]

            X_train = train_df[selected_features]
            y_train = train_df["is_fraud"]

            X_test = test_df[selected_features]
            y_test = test_df["is_fraud"]

            numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
            categorical_features = X_train.select_dtypes(include=["object"]).columns

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_features),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                ]
            )

            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            os.makedirs(os.path.dirname(self.config.preprocessor_path), exist_ok=True)
            joblib.dump(preprocessor, self.config.preprocessor_path)

            logging.info("Data transformation completed")

            return X_train_processed, y_train, X_test_processed, y_test

        except Exception as e:
            raise CustomException(e, sys)