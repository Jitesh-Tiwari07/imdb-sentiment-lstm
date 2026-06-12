import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

MAX_FEATURES = 10000
MAX_LEN = 200

def load_and_preprocess_data():
    print("Loading IMDB dataset...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)
    
    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples:  {len(x_test)}")
    
    print("Padding sequences...")
    x_train = pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = pad_sequences(x_test, maxlen=MAX_LEN)
    
    return (x_train, y_train), (x_test, y_test)

def get_word_index():
    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    return word_index

def preprocess_review(review_text, word_index, max_len=MAX_LEN):
    tokens = [1]  # <START>
    for word in review_text.lower().split():
        tokens.append(word_index.get(word, 2))  # <UNK>
    
    padded = pad_sequences([tokens], maxlen=max_len)
    return padded