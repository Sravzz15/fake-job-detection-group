# JobCheck – Detecting Fake Job Posts Using NLP (Group)

## Overview :
This project detects fake or fraudulent job postings using Natural Language Processing (NLP) techniques. It demonstrates a full pipeline: data ingestion, preprocessing, feature engineering (text vectorization), model training and evaluation, and a simple web API for inference. The goal is to provide an explainable, reproducible baseline you can extend and deploy.

## Features :
Data cleaning and text preprocessing (tokenization, lowercasing, stopword removal, lemmatization)
TF-IDF and/or count-vector text features
Baseline classifiers (Logistic Regression, Random Forest, SVM) and example of model selection
Model evaluation with accuracy, precision, recall, F1-score, ROC AUC, and confusion matrix
Simple Flask API for serving predictions

## Tech Stack :
- FrontEnd : HTML, CSS, JavaScript.
- Backend : Flask, Python.
- DataBase : SQLite.

## Step to Setup Project in Local Devices
Step 1: Clone this repository
```bash
   git clone https://github.com/AnilSonawane2/fake-job-detection-group.git
```

Step 2: Change the Directory
```bash
   cd fake-job-detection-group
```

Step 3: Create your Virtual Environment
```bash
   python -m venv venv
```

Step 4: Activate Virtual Environment
```bash
   .\venv\Scripts\activate
```

Step 5: Install all required libraries
```bash
   pip intall -r requirements.txt
```
Step 6: Run the Flask App
```bash
   python app.py
```
