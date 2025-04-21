from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import re
import os

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

app = FastAPI(title="Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SentimentRequest(BaseModel):
    text: str

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    data = {
        'text': [
            "I love this product, it's amazing!",
            "This is the best movie I've ever watched",
            "The service was excellent and staff very friendly",
            "I hate this, it's terrible",
            "Worst experience ever, do not recommend",
            "The quality is very poor and it broke after one use",
            "Pretty good product but could be better",
            "Not the best but not the worst either"
        ],
        'sentiment': ['positive', 'positive', 'positive', 'negative', 'negative', 'negative', 'neutral', 'neutral']
    }
    df = pd.DataFrame(data)
    
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Model accuracy: {model.score(X_test, y_test)}")
else:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print("Loaded existing model and vectorizer")

@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API", "status": "online"}

@app.post("/analyze")
async def analyze_sentiment(request: SentimentRequest):
    try:
        processed_text = preprocess_text(request.text)
        
        text_vector = vectorizer.transform([processed_text])
        
        sentiment = model.predict(text_vector)[0]
        
        probabilities = model.predict_proba(text_vector)[0]
        prob_dict = {class_name: float(prob) for class_name, prob in zip(model.classes_, probabilities)}
        
        return JSONResponse(content={
            "text": request.text,
            "sentiment": sentiment,
            "confidence_scores": prob_dict
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
