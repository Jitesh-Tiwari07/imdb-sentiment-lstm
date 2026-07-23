import streamlit as st
import tensorflow as tf
import os
from utils import get_word_index, preprocess_review

# ====================== CONFIG ======================
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎥",
    layout="centered"
)

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/Best_LSTM_IMDB.keras')

model = load_model()
word_index = get_word_index()

# ====================== SESSION STATE ======================
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# Callback function to update review from example buttons
def set_review(example_text):
    st.session_state.review_text = example_text

# ====================== SIDEBAR ======================
st.sidebar.title("🎬 About")
st.sidebar.info(
    "LSTM-based Sentiment Analysis trained on IMDB dataset.\n\n"
    "Predict whether a movie review is **Positive** or **Negative**."
)

st.sidebar.metric("Model Accuracy", "86.5%")
st.sidebar.metric("Vocabulary Size", "10,000")
st.sidebar.metric("Max Review Length", "200 words")

# ====================== MAIN APP ======================
st.title("🎥 IMDB Sentiment Analyzer")
st.markdown("### LSTM + Embedding Model")

# Text Area - Controlled by session state
review = st.text_area(
    "Write your movie review here:",
    value=st.session_state.review_text,
    height=150,
    placeholder="This movie was absolutely fantastic! The acting and story were brilliant...",
    key="review_input"
)

# Example Buttons
st.markdown("### Quick Examples")
example_btns = [
    "This movie was absolutely fantastic with brilliant acting!",
    "Worst film I have ever seen. Complete waste of time.",
    "The story was okay but the ending was disappointing.",
    "Not bad at all, I really enjoyed it. Great visuals!"
]

cols = st.columns(2)
for i, ex in enumerate(example_btns):
    with cols[i % 2]:
        if st.button(ex[:55] + "..." if len(ex) > 55 else ex, key=f"btn_{i}"):
            set_review(ex)
            st.rerun()

# ====================== PREDICTION ======================
if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if not review or review.strip() == "":
        st.warning("Please enter a review!")
    else:
        with st.spinner("Analyzing sentiment..."):
            padded = preprocess_review(review, word_index)
            prediction = model.predict(padded, verbose=0)[0][0]
            
            sentiment = "Positive" if prediction > 0.5 else "Negative"
            confidence = float(prediction)
            
            # Display Result
            if sentiment == "Positive":
                st.success(f"✅ **{sentiment}** Review", icon="😊")
                st.progress(confidence)
                st.metric("Positive Confidence", f"{confidence*100:.1f}%")
            else:
                st.error(f"❌ **{sentiment}** Review", icon="😞")
                st.progress(confidence)
                st.metric("Positive Confidence", f"{confidence*100:.1f}%")

# ====================== HISTORY ======================
if 'history' not in st.session_state:
    st.session_state.history = []

if review and st.button("💾 Save to History", use_container_width=True):
    if review.strip():
        padded = preprocess_review(review, word_index)
        pred = model.predict(padded, verbose=0)[0][0]
        sentiment = "Positive" if pred > 0.5 else "Negative"
        
        st.session_state.history.append({
            "review": review[:150] + "..." if len(review) > 150 else review,
            "sentiment": sentiment,
            "confidence": float(pred)
        })
        st.success("Saved to history!")

if st.session_state.history:
    st.markdown("### 📜 Recent Predictions")
    for entry in reversed(st.session_state.history[-5:]):
        emoji = "😊" if entry["sentiment"] == "Positive" else "😞"
        st.write(f"{emoji} **{entry['sentiment']}** ({entry['confidence']:.3f}) — {entry['review']}")

# ====================== FOOTER ======================
st.markdown("---")
st.markdown("Built with **TensorFlow LSTM** & Streamlit | Learning Project 🎉")
