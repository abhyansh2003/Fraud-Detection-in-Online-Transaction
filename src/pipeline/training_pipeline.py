from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
    def start(self, source_file_path):
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion(source_file_path)

        validation = DataValidation()
        validation.initiate_data_validation(train_path)

        transformation = DataTransformation()
        X_train, y_train, X_test, y_test = transformation.initiate_data_transformation(
            train_path, test_path
        )

        trainer = ModelTrainer()
        score = trainer.initiate_model_training(X_train, y_train, X_test, y_test)

        print(f"Training Completed. ROC AUC: {score}")