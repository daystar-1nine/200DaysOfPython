
import tensorflow as tf

def get_callbacks(model_name="best_model.keras"):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name, monitor="val_auc", mode="max", save_best_only=True)
    return [early_stopping, reduce_lr, checkpoint]

def train_model(model, X_train, y_train, X_val, y_val, optimizer_name='adam', lr=0.001, batch_size=32):
    if optimizer_name == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=batch_size,
        callbacks=get_callbacks(),
        verbose=0
    )
    return model, history
