import os
import sys
import pandas as pd
import yaml
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataValidationConfig:
    schema_file_path: str = "src/config/schema.yaml"


class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()

    def validate_columns(self, df: pd.DataFrame):
        try:
            with open(self.validation_config.schema_file_path, "r") as file:
                schema = yaml.safe_load(file)

            expected_columns = list(schema["columns"].keys())

            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                raise Exception(f"Missing columns: {missing_cols}")

            logging.info("All required columns are present")

        except Exception as e:
            raise CustomException(e, sys)

    def validate_nulls(self, df: pd.DataFrame):
        try:
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                logging.warning(f"Null values detected:\n{null_counts}")
            else:
                logging.info("No null values found")

        except Exception as e:
            raise CustomException(e, sys)

    def validate_target(self, df: pd.DataFrame):
        try:
            if not set(df["is_fraud"].unique()).issubset({0, 1}):
                raise Exception("Target column must contain only 0 and 1")

            logging.info("Target column validated")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self, file_path: str):
        try:
            df = pd.read_csv(file_path)

            self.validate_columns(df)
            self.validate_nulls(df)
            self.validate_target(df)

            logging.info("Data validation completed successfully")
            return True

        except Exception as e:
            raise CustomException(e, sys)