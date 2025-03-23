from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')  # Important based on your preprocessing

# ===== Load Saved Components =====
svm_model = joblib.load("models/svm_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
ratios = joblib.load("models/nb_ratios.joblib")

# ===== Initialize FastAPI =====
app = FastAPI()

# ===== CORS Middleware =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:5173"] if you want to restrict it
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Preprocessing Setup =====
stop_words_set = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def handle_negations(text):
    return re.sub(r"\b(not|n't)\s+(\w+)", r"not_\2", text)

def advanced_data_cleaning(text):
    text = text.lower()
    text = handle_negations(text)
    text = re.sub('<br />', '', text)
    text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", '', text)
    tokens = word_tokenize(text)
    filtered = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words_set]
    return " ".join(filtered)

# ===== Request Body Schema =====
class Review(BaseModel):
    text: str

# ===== Prediction Endpoint =====
@app.post("/predict")
def predict_sentiment(review: Review):
    cleaned = advanced_data_cleaning(review.text)
    tfidf = vectorizer.transform([cleaned])
    tfidf_nb = tfidf.multiply(ratios)
    prediction = svm_model.predict(tfidf_nb)[0]
    sentiment = "positive" if prediction == 1 else "negative"
    return {"sentiment": sentiment}
