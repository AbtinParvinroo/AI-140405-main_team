from tensorflow.keras.models import load_model
import pandas as pd
import logging
import joblib
import os

logger = logging.getLogger("SystemIO")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)

class SystemIO:
    def __init__(self, save_dir: str = './models'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save_df(self, df: pd.DataFrame, name: str) -> str:
        safe_name = os.path.basename(name)
        path = os.path.join(self.save_dir, f"{safe_name}.parquet")
        df.to_parquet(path)
        logger.info(f"DataFrame saved to {path}")
        return path

    def load_df(self, name: str) -> pd.DataFrame:
        safe_name = os.path.basename(name)
        path = os.path.join(self.save_dir, f"{safe_name}.parquet")
        df = pd.read_parquet(path)
        logger.info(f"DataFrame loaded from {path}")
        return df

    def save_keras(self, model, name: str) -> str:
        safe_name = os.path.basename(name)
        path = os.path.join(self.save_dir, f"{safe_name}.keras")  # استفاده از فرمت جدید
        model.save(path)
        logger.info(f"Keras model saved to {path}")
        return path

    def load_keras(self, name: str):
        safe_name = os.path.basename(name)
        path = os.path.join(self.save_dir, f"{safe_name}.keras")
        model = load_model(path)
        logger.info(f"Keras model loaded from {path}")
        return model

    def save_dict(self, obj: dict, name: str) -> str:
        safe_name = os.path.basename(name)
        path = os.path.join(self.save_dir, f"{safe_name}.joblib")
        joblib.dump(obj, path)
        logger.info(f"Dict saved to {path}")
        return path

    def load_dict(self, name: str) -> dict:
        safe_name = os.path.basename(name)
        path = os.path.join(self.save_dir, f"{safe_name}.joblib")
        obj = joblib.load(path)
        logger.info(f"Dict loaded from {path}")
        return obj