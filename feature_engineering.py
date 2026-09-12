import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os

print("=" * 60)
print("FEATURE ENGINEERING STARTED")
print("=" * 60)

# Load data
print("\nLoading cleaned dataset...")
df = pd.read_csv('dataset/cleaned_dataset.csv')
print(f"Loaded {len(df)} rows")

# Features and labels
X = df['clean_statement']
y = df['Label']

# TF-IDF
print("\nConverting text to numbers...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X)
print(f"TF-IDF shape: {X_tfidf.shape}")

# Train-test split
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# Create models folder
if not os.path.exists('models'):
    os.makedirs('models')

# Save
print("\nSaving artifacts...")
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
joblib.dump(X_train, 'models/X_train.pkl')
joblib.dump(X_test, 'models/X_test.pkl')
joblib.dump(y_train, 'models/y_train.pkl')
joblib.dump(y_test, 'models/y_test.pkl')
print("Saved all files to 'models/' folder")

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE!")
print("=" * 60)