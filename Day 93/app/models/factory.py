
import tensorflow as tf

def build_dense_baseline():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

def build_custom_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

def build_resnet_model():
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=(32, 32, 3)
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

def build_mobilenet_model():
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(32, 32, 3)
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)
    
def mock_params_and_size(name):
    if name == 'Dense Baseline': return 394000, 1.5
    if name == 'Custom CNN': return 2100000, 8.0
    if name == 'ResNet50': return 23000000, 92.0
    if name == 'MobileNetV2': return 2200000, 8.8
    return 0, 0.0
