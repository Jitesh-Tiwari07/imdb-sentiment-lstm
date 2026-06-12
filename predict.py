import tensorflow as tf
from utils import get_word_index, preprocess_review

# Load model (change path if needed)
model = tf.keras.models.load_model('models/Best_LSTM_IMDB.keras')
word_index = get_word_index()

def predict_sentiment(review_text):
    padded = preprocess_review(review_text, word_index)
    prediction = model.predict(padded, verbose=0)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, float(prediction)

if __name__ == "__main__":
    reviews = [
        "This movie was absolutely fantastic with brilliant acting!",
        "Worst film I have ever seen. Complete waste of time.",
        "The story was okay but the ending was disappointing.",
        "Not bad at all, I really enjoyed it."
    ]

    print("\n=== Custom Review Predictions ===")
    for rev in reviews:
        sentiment, score = predict_sentiment(rev)
        print(f"Review: {rev}")
        print(f"Prediction: {sentiment} (Confidence: {score:.4f})\n")