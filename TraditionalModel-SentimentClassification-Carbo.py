import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Download IMDb dataset CSV (e.g., from https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
# Load dataset from CSV
df = pd.read_csv("IMDB Dataset.csv")  # Replace with your file path

# Preprocess text
def preprocess(text):
    text = text.lower()
    text = "".join([c for c in text if c.isalnum() or c == " "])
    return text

df["cleaned_text"] = df["review"].apply(preprocess)

# Convert labels to binary (0 = negative, 1 = positive)
df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1})

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_text"], df["sentiment"], test_size=0.2, random_state=42
)

# TF-IDF Vectorization (fit only on training data)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)  # Transform test data

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = lr_model.predict(X_test_tfidf)

print("Traditional Model Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")