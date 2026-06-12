# 🎬 IMDB Sentiment Analysis using LSTM

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Project Overview

This project implements a Long Short-Term Memory (LSTM) based Deep Learning model for sentiment analysis on movie reviews from the IMDB dataset.

The model classifies reviews as Positive or Negative by learning contextual patterns and word dependencies in text data. Advanced regularization techniques such as Dropout, L2 Regularization, Early Stopping, and Learning Rate Scheduling are incorporated to improve generalization and reduce overfitting.

A Streamlit web application is also included for real-time sentiment prediction on custom user reviews.

---

## 🚀 Features

✅ IMDB Movie Review Sentiment Classification

✅ Deep Learning using LSTM Networks

✅ Word Embedding Representation

✅ Dropout & L2 Regularization

✅ Early Stopping for Overfitting Prevention

✅ Learning Rate Scheduling

✅ Model Checkpointing

✅ Real-Time Custom Review Prediction

✅ Interactive Streamlit Web Interface

✅ Accuracy & Loss Visualization

---

## 🧠 Model Architecture

Input Layer (200 Tokens)

⬇

Embedding Layer (128 Dimensions)

⬇

LSTM Layer (64 Units)

⬇

Dropout Layer (50%)

⬇

Dense Output Layer (Sigmoid)

---

## 📂 Project Structure

```text
IMDB-Sentiment-Analysis-LSTM/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── Best_LSTM_IMDB.keras
│
├── images/
│   ├── accuracy_loss_graph.png
│   └── app_screenshot.png
│
└── notebooks/
    └── sentiment_analysis.ipynb
```

---

## 📊 Dataset

Dataset: IMDB Movie Reviews Dataset

* 50,000 Movie Reviews
* Binary Sentiment Classification
* Positive Reviews → 1
* Negative Reviews → 0

Training Samples: 25,000

Testing Samples: 25,000

---

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Streamlit
* Deep Learning
* Natural Language Processing (NLP)

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/IMDB-Sentiment-Analysis-LSTM.git
cd IMDB-Sentiment-Analysis-LSTM
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

```bash
python train_model.py
```

The best model will automatically be saved inside:

```text
models/Best_LSTM_IMDB.keras
```

---

## 🌐 Run the Streamlit Application

```bash
streamlit run app.py
```

Open your browser and visit:

```text
http://localhost:8501
```

---

## 📈 Training Performance

The model tracks:

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss

Performance graphs are generated automatically after training.

---

## 🎯 Sample Predictions

| Review                               | Prediction |
| ------------------------------------ | ---------- |
| This movie was absolutely fantastic! | Positive   |
| Worst film I have ever seen.         | Negative   |
| Not bad at all, I enjoyed it.        | Positive   |
| The story was disappointing.         | Negative   |

---

## 💡 Key Learning Outcomes

* Text Preprocessing for NLP
* Sequence Padding
* Word Embeddings
* LSTM Networks
* Binary Classification
* Model Regularization
* Hyperparameter Tuning
* Streamlit Deployment

---

## 📸 Application Preview


<img width="1911" height="978" alt="image" src="https://github.com/user-attachments/assets/9a18366d-be53-4e23-b3f7-5968d3973258" />


<img width="1492" height="614" alt="image" src="https://github.com/user-attachments/assets/7e417b33-bd0f-4b8f-8337-d31ed68fb6ec" />
(images/accuracy_loss_graph.png)
---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

Your support motivates further development and improvements.
