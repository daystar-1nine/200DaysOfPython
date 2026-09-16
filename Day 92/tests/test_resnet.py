
import pytest
import numpy as np
import tensorflow as tf
from app.data.loader import load_and_preprocess_resnet
from app.models.factory import build_resnet_model

def test_data_resnet_preprocess():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_resnet()
    assert len(x_train.shape) == 4
    assert x_train.shape[-1] == 3 # Converted to RGB

def test_frozen_resnet():
    model, base = build_resnet_model(trainable_layers=0)
    assert base.trainable == False

def test_fine_tuning_resnet():
    model, base = build_resnet_model(trainable_layers=20)
    assert base.trainable == True
