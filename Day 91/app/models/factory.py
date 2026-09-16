
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

def build_transfer_model(trainable_layers=0, dropout_rate=0.3):
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(32, 32, 3)
    )
    
    # Freeze logic
    base_model.trainable = False
    
    if trainable_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
            
    inputs = tf.keras.Input(shape=(32, 32, 3))
    
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10)
    ])
    
    x = augmentation(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model, base_model
