# 🧠 Sentiment Analysis API with FastAPI

This project is a lightweight **Sentiment Analysis API** built using **FastAPI**, designed to predict the sentiment (Positive, Negative, or Neutral) of a given text input. The model is trained on a small sample dataset and served via an interactive and production-ready API.

---

## 🚀 Features

- ✅ RESTful API using FastAPI
- 🧹 Preprocessing with NLTK (tokenization, stopword removal)
- 🧠 Machine Learning using Logistic Regression and TF-IDF
- 📊 Returns sentiment and confidence scores
- 🔄 Auto-reloads with Uvicorn for local development
- 🌐 CORS support for cross-origin requests

---

## 📦 Requirements

```bash
pip install fastapi uvicorn scikit-learn pandas nltk pydantic
