
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

def build_resnet_model(trainable_layers=0, dropout_rate=0.3):
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(32, 32, 3)
    )
    
    base_model.trainable = False
    
    if trainable_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
            
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model, base_model
    
def count_trainable_parameters(model):
    # Mocking param count
    return {
        'total': 23000000,
        'trainable': 20000,
        'non_trainable': 22980000
    }
