import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from utils import load_and_preprocess_data

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Hyperparameters
MAX_FEATURES = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 64
EPOCHS = 10

def build_model():
    model = Sequential()
    model.add(Input(shape=(MAX_LEN,)))
    model.add(
        Embedding(
            input_dim=MAX_FEATURES,
            output_dim=EMBEDDING_DIM,
            mask_zero=True
        )
    )
    model.add(
        LSTM(
            units=LSTM_UNITS,
            dropout=0.5,
            recurrent_dropout=0.3,
            kernel_regularizer=l2(0.001),
            recurrent_regularizer=l2(0.001)
        )
    )
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    model = build_model()
    print("\nMODEL STRUCTURE")
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(
        filepath='models/Best_LSTM_IMDB.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1
    )

    print("\nStarting Training...")
    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[checkpoint, early_stop, lr_reducer],
        verbose=1
    )

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('plots/training_history.png')
    plt.show()

    # Evaluate best model
    best_model = tf.keras.models.load_model('models/Best_LSTM_IMDB.keras')
    test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
    
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
    print(f"Final Test Loss    : {test_loss:.4f}")

if __name__ == "__main__":
    main()