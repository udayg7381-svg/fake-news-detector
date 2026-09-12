import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🤖 TRAINING WITH FIXED BALANCED DATASET")
print("="*60)

# Load fixed dataset
print("\n📁 Loading fixed dataset...")
df = pd.read_csv('dataset/final_balanced_dataset.csv')
X = df['clean_statement']
y = df['Label']
print(f"✅ Loaded {len(df)} rows")
print(f"Real: {sum(y==0)}, Fake: {sum(y==1)}")

# TF-IDF
print("\n🔄 Converting text to numbers...")
vectorizer = TfidfVectorizer(
    max_features=7000,
    stop_words='english',
    ngram_range=(1, 2)
)
X_tfidf = vectorizer.fit_transform(X)
print(f"✅ TF-IDF shape: {X_tfidf.shape}")

# Split
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Training: {X_train.shape[0]} samples")
print(f"✅ Test: {X_test.shape[0]} samples")

# Train
print("\n🚀 Training Logistic Regression...")
model = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"   ✅ Accuracy: {accuracy:.4f}")
print(f"   ✅ Precision: {precision:.4f}")
print(f"   ✅ Recall: {recall:.4f}")
print(f"   ✅ F1-Score: {f1:.4f}")

print(f"\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save
print("\n💾 Saving final correct model...")
joblib.dump(model, 'models/final_correct_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer_correct.pkl')
print("✅ Saved as 'models/final_correct_model.pkl'")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)