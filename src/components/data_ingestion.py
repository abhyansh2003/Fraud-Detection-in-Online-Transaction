import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = "artifacts/data_ingestion/raw.csv"
    train_data_path: str = "artifacts/data_ingestion/train.csv"
    test_data_path: str = "artifacts/data_ingestion/test.csv"


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, source_file_path: str):
        logging.info("Entered data ingestion method")

        try:
            # Create directory if not exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Read dataset
            df = pd.read_csv(source_file_path)
            logging.info("Read dataset successfully")

            # Save raw copy
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Splitting dataset into train and test")

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["is_fraud"]  # Important for imbalance
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)