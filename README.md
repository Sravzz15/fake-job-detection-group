# fake-job-detection-group

## Overview :
This project detects fake or fraudulent job postings using Natural Language Processing (NLP) techniques. It demonstrates a full pipeline: data ingestion, preprocessing, feature engineering (text vectorization), model training and evaluation, and a simple web API for inference. The goal is to provide an explainable, reproducible baseline you can extend and deploy.
## Features :
Data cleaning and text preprocessing (tokenization, lowercasing, stopword removal, lemmatization)
TF-IDF and/or count-vector text features
Baseline classifiers (Logistic Regression, Random Forest, SVM) and example of model selection
Model evaluation with accuracy, precision, recall, F1-score, ROC AUC, and confusion matrix
Simple Flask API for serving predictions
