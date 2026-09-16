
import pytest
import numpy as np
import tensorflow as tf
from app.data.loader import load_and_preprocess
from app.models import build_cnn_dropout

def test_data_loader():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess()
    assert len(x_train.shape) == 4  # (N, H, W, C)
    assert x_train.shape[3] == 1    # 1 channel

def test_model_build():
    model = build_cnn_dropout()
    assert len(model.layers) > 5
