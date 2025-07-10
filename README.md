# 🎯 Sentiment Classification with LSTM + Custom Attention (IMDB Dataset)

This project builds a binary text classification model using **TensorFlow/Keras**, combining an **LSTM network** with a **custom attention mechanism** to predict movie review sentiment (positive or negative) from the IMDB dataset.

---
## Install the pip
pip install tensorflow numpy

## 📌 Overview

- 📚 Dataset: IMDB Movie Reviews (binary labels: positive or negative)
- 🧠 Model: Embedding → LSTM → Attention Layer → Dense (sigmoid)
- 🛠️ Framework: TensorFlow / Keras
- ✅ Task: Binary sentiment classification

---
## 🚀 Model Architecture

```text
Input Layer (shape: [200])
    ↓
Embedding Layer (output: [200, 128])
    ↓
LSTM Layer (64 units, return_sequences=True)
    ↓
Custom Attention Layer (learns weighted importance of each word)
    ↓
Dense Layer with Sigmoid (output: [1] → sentiment probability)


---

## 📂 How to Run

### 1. Clone this repo
```bash
git clone https://github.com/your-username/imdb-lstm-attention.git
cd imdb-lstm-attention
