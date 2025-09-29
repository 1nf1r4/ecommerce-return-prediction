import pandas as pd
import logging
from typing import Dict, Any
from datetime import datetime
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineeringAgent:
    """
    Feature Engineering Agent
    Purpose: Create derived features from raw/preprocessed inputs
    """

    def __init__(self, scaler_path=None, encoder_path=None):
        self.required_columns = [
            "Product_Price", "Order_Quantity", "User_Age", 
            "User_Location", "Discount_Applied", "Order_Date"
        ]
        self.scaler = None
        self.encoder = None
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        if encoder_path:
            with open(encoder_path, 'rb') as f:
                self.encoder = pickle.load(f)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations
        Args:
            df: Preprocessed dataframe
        Returns:
            DataFrame with new engineered features
        """
        logger.info("Starting feature engineering...")

        # Ensure required columns exist
        for col in self.required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Type safety for numeric columns
        df["Product_Price"] = pd.to_numeric(df["Product_Price"], errors="coerce")
        df["Order_Quantity"] = pd.to_numeric(df["Order_Quantity"], errors="coerce")
        df["Discount_Applied"] = pd.to_numeric(df["Discount_Applied"], errors="coerce")

        # 1. Total Order Value
        df["Total_Order_Value"] = df["Product_Price"] * df["Order_Quantity"]

        # 2. Temporal Features
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
        df["Order_Year"] = df["Order_Date"].dt.year
        df["Order_Month"] = df["Order_Date"].dt.month
        df["Order_Weekday"] = df["Order_Date"].dt.weekday
        # Warn if any dates could not be parsed
        if df["Order_Date"].isnull().any():
            logger.warning("Some Order_Date values could not be parsed and are set as NaT.")

        # 3. Encode User Location using fitted encoder if available
        if self.encoder:
            try:
                df["User_Location_Num"] = self.encoder.transform(df[["User_Location"]])
            except Exception as e:
                logger.warning(f"Encoding User_Location failed: {e}")
                df["User_Location_Num"] = -1
        else:
            location_map = {"Urban": 1, "Rural": 0, "Suburban": 2}
            df["User_Location_Num"] = df["User_Location"].map(location_map)
            df["User_Location_Num"] = df["User_Location_Num"].fillna(-1)
            if (df["User_Location_Num"] == -1).any():
                logger.warning("Some User_Location values are unknown and encoded as -1.")

        # 4. Discount transformations (robust)
        df["Discount_Amount"] = df["Product_Price"] * (df["Discount_Applied"].fillna(0) / 100)
        df["Effective_Price"] = df["Product_Price"] - df["Discount_Amount"]
        # If discount is negative or >100, warn
        invalid_discount = (df["Discount_Applied"] < 0) | (df["Discount_Applied"] > 100)
        if invalid_discount.any():
            logger.warning("Some Discount_Applied values are outside [0, 100].")

        # 5. Apply fitted scaler if available (for numeric columns)
        if self.scaler:
            num_cols = ["Product_Price", "Order_Quantity", "Discount_Applied", "Total_Order_Value", "Discount_Amount", "Effective_Price", "User_Age"]
            try:
                df[num_cols] = self.scaler.transform(df[num_cols])
            except Exception as e:
                logger.warning(f"Scaling failed: {e}")

        logger.info("Feature engineering completed successfully.")
        return df

# Example usage
if __name__ == "__main__":
    # Sample preprocessed data
    sample_data = pd.DataFrame([{
        "Product_Price": 1500,
        "Order_Quantity": 1,
        "User_Age": 24,
        "User_Location": "Urban",
        "Discount_Applied": 2,
        "Order_Date": "2024-01-15"
    }])

    agent = FeatureEngineeringAgent()
    result_df = agent.transform(sample_data)
    print(result_df)
