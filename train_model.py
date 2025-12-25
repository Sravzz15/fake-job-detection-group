import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
import json # Added for Task 1

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_DIR, 'fake_job_postings.csv')
MODEL_FILE = os.path.join(BASE_DIR, 'fake_job_model.pkl')
VECTORIZER_FILE = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')

# Download Resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def clean_text(text):
    if pd.isnull(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

def train():
    # 1. Load Data
    if not os.path.exists(DATASET_FILE):
        print(json.dumps({"error": "Dataset not found"}))
        return

    df = pd.read_csv(DATASET_FILE)
    
    # 2. Preprocessing
    df.fillna(' ', inplace=True)
    df['text'] = df['title'] + " " + df['company_profile'] + " " + df['description'] + " " + df['requirements']
    df['clean_text'] = df['text'].apply(clean_text)

    # 3. Vectorization
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X = tfidf.fit_transform(df['clean_text'])
    y = df['fraudulent']

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ==========================================
    # TASK 1: CHECK OLD MODEL PERFORMANCE FIRST
    # ==========================================
    old_accuracy = 0.0
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
            old_model = joblib.load(MODEL_FILE)
            old_vectorizer = joblib.load(VECTORIZER_FILE)
            
           
            try:
                
                # if vectorizer vocab changed, or 88.4% as a baseline simulation if it fails.
                y_pred_old = old_model.predict(X_test) 
                old_accuracy = accuracy_score(y_test, y_pred_old)
            except:
                old_accuracy = 0.0
    except Exception:
        old_accuracy = 0.0

    # ==========================================
    # TRAIN NEW MODEL
    # ==========================================
    new_model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
    new_model.fit(X_train, y_train)

    # Evaluate New Model
    y_pred_new = new_model.predict(X_test)
    new_accuracy = accuracy_score(y_test, y_pred_new)

    # Save New Model
    joblib.dump(new_model, MODEL_FILE)
    joblib.dump(tfidf, VECTORIZER_FILE)

    # ==========================================
    # PREPARE DATA FOR APP.PY (TASK 1 & 2)
    # ==========================================
    results = {
        "old_accuracy": round(old_accuracy * 100, 1),
        "new_accuracy": round(new_accuracy * 100, 1),
        "improvement": round((new_accuracy - old_accuracy) * 100, 1),
        "record_count": len(df)
    }

    # Print JSON between markers so app.py can parse it safely
    print("JSON_START")
    print(json.dumps(results))
    print("JSON_END")

if __name__ == "__main__":
    train()