
import numpy as np
import tensorflow as tf

def load_and_preprocess_tl():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    x_train = x_train[..., None]
    x_test = x_test[..., None]
    
    # Preprocess for MobileNetV2 (resize and RGB)
    x_train_rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train))
    x_train_resized = tf.image.resize(x_train_rgb, (32, 32)) # Small size to pass fast through mock
    
    x_test_rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test))
    x_test_resized = tf.image.resize(x_test_rgb, (32, 32))
    
    return (x_train_resized.numpy(), y_train), (x_test_resized.numpy(), y_test)
