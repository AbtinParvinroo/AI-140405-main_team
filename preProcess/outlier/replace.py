import pandas as pd
import numpy as np

class MaskFiller:
    def __init__(self, mask: pd.DataFrame, df, features, verbose=False):
        self.df = df.copy()
        self.features = features
        self.verbose = verbose
        self.mask = mask

    def fill_ffill(self):
        df_filled = self.df.copy()
        for col in self.features:
            if col not in self.mask.columns:
                raise ValueError(f"Column {col} not in mask")

            df_filled.loc[self.mask[col], col] = np.nan
            df_filled[col] = df_filled[col].ffill()
            if self.verbose:
                print(f"{col} filled with forward fill")

        return df_filled

    def fill_bfill(self):
        df_filled = self.df.copy()
        for col in self.features:
            if col not in self.mask.columns:
                raise ValueError(f"Column {col} not in mask")

            df_filled.loc[self.mask[col], col] = np.nan
            df_filled[col] = df_filled[col].bfill()
            if self.verbose:
                print(f"{col} filled with backward fill")

        return df_filled