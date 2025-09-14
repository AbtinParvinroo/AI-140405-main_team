from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
from save_load import SystemIO
import pandas as pd
import pytest
import os

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9]
    })

@pytest.fixture
def sample_dict():
    return {"x": 10, "y": [1,2,3]}

@pytest.fixture
def sample_keras_model():
    model = Sequential([
        Input(shape=(3,)),
        Dense(4, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def test_save_load_df(tmp_dir, sample_df):
    io = SystemIO(save_dir=tmp_dir)
    path = io.save_df(sample_df, "df_test")
    assert os.path.exists(path)
    df_loaded = io.load_df("df_test")
    pd.testing.assert_frame_equal(df_loaded, sample_df)

def test_save_load_dict(tmp_dir, sample_dict):
    io = SystemIO(save_dir=tmp_dir)
    path = io.save_dict(sample_dict, "dict_test")
    assert os.path.exists(path)
    dict_loaded = io.load_dict("dict_test")
    assert dict_loaded == sample_dict

def test_save_load_keras(tmp_dir, sample_keras_model):
    io = SystemIO(save_dir=tmp_dir)
    path = io.save_keras(sample_keras_model, "keras_test")
    assert os.path.exists(path)
    loaded_model = io.load_keras("keras_test")
    assert len(loaded_model.layers) == len(sample_keras_model.layers)