import joblib
import pandas as pd


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load("artifacts/model_trainer/model.pkl")
        self.threshold = joblib.load("artifacts/model_trainer/threshold.pkl")
        self.preprocessor = joblib.load("artifacts/data_transformation/preprocessor.pkl")
        self.required_columns = self.preprocessor.feature_names_in_.tolist()

    def predict(self, input_df):
        # Add missing columns with default value 0
        for col in self.required_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Ensure correct column order
        input_df = input_df[self.required_columns]
        
        # Feature Engineering during prediction

        input_df["amount_ratio"] = input_df["amount"] / (input_df["avg_amount_last_24h"] + 1)

        input_df["high_amount_flag"] = (input_df["amount"] > 10000).astype(int)

        input_df["night_transaction"] = (input_df["hour_of_day"] <= 5).astype(int)
        
        processed_data = self.preprocessor.transform(input_df)
        probs = self.model.predict_proba(processed_data)[:, 1]
        preds = (probs >= 0.25).astype(int)
        return preds, probs
    
    