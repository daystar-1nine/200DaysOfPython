
import pytest
import numpy as np
import tensorflow as tf
from app.data.loader import load_and_preprocess_eff
from app.models.factory import build_mobilenet_model

def test_data_preprocess():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_eff()
    assert len(x_train.shape) == 4
    assert x_train.shape[-1] == 3 

def test_mobilenet():
    model = build_mobilenet_model()
    assert model is not None
