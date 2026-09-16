
import pytest
import numpy as np
import tensorflow as tf
from app.data.loader import load_and_preprocess_tl
from app.models.factory import build_transfer_model

def test_data_preprocessing_rgb():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_tl()
    assert len(x_train.shape) == 4
    assert x_train.shape[-1] == 3 # Converted to RGB

def test_frozen_backbone():
    model, base = build_transfer_model(trainable_layers=0)
    assert base.trainable == False

def test_fine_tuning_unfreezes():
    model, base = build_transfer_model(trainable_layers=10)
    assert base.trainable == True
