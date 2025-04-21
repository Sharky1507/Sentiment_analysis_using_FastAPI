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

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

app = FastAPI(title="Sentiment Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request model
class SentimentRequest(BaseModel):
    text: str

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back to string
    return ' '.join(tokens)

# Define paths for model and vectorizer
MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# Check if model exists, otherwise train a new one
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    # Sample dataset - in a real scenario, you would load your actual dataset
    # This is just a dummy dataset for demonstration
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
    
    # Preprocess the text data
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['sentiment']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Save the model and vectorizer
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Model accuracy: {model.score(X_test, y_test)}")
else:
    # Load the trained model and vectorizer
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
        # Preprocess the input text
        processed_text = preprocess_text(request.text)
        
        # Transform the text using the TF-IDF vectorizer
        text_vector = vectorizer.transform([processed_text])
        
        # Make prediction
        sentiment = model.predict(text_vector)[0]
        
        # Get probability scores for each class
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