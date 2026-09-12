import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import time
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🔧 FINAL MODEL TRAINING (With Hyperparameter Tuning)")
print("="*60)

# Load data
print("\n📁 Loading cleaned dataset...")
df = pd.read_csv('dataset/cleaned_dataset.csv')
X = df['clean_statement']
y = df['Label']
print(f"✅ Loaded {len(df)} rows")

# TF-IDF with more features
print("\n🔄 Converting text to numbers with more features...")
vectorizer = TfidfVectorizer(
    max_features=7000,  # Increased from 5000
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)
X_tfidf = vectorizer.fit_transform(X)
print(f"✅ TF-IDF shape: {X_tfidf.shape}")

# Train-Test Split
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Training: {X_train.shape[0]} samples")
print(f"✅ Test: {X_test.shape[0]} samples")

# SMOTE
print("\n⚖️ Applying SMOTE...")
print(f"Before SMOTE - Real: {sum(y_train==0)}, Fake: {sum(y_train==1)}")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"After SMOTE - Real: {sum(y_train_balanced==0)}, Fake: {sum(y_train_balanced==1)}")

# Hyperparameter Tuning
print("\n🔧 Hyperparameter Tuning...")
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['liblinear', 'saga'],
    'max_iter': [1000, 2000]
}

logreg = LogisticRegression(random_state=42)
grid_search = GridSearchCV(
    logreg, param_grid, cv=3, scoring='f1', n_jobs=-1
)
grid_search.fit(X_train_balanced, y_train_balanced)
best_model = grid_search.best_estimator_
print(f"✅ Best Parameters: {grid_search.best_params_}")

# Predict
print("\n📊 Evaluating best model...")
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"   ✅ Accuracy: {accuracy:.4f}")
print(f"   ✅ Precision: {precision:.4f}")
print(f"   ✅ Recall: {recall:.4f}")
print(f"   ✅ F1-Score: {f1:.4f}")

# Save final model
print("\n💾 Saving final model...")
joblib.dump(best_model, 'models/best_model_final.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer_final.pkl')
print("✅ Saved as 'models/best_model_final.pkl'")

print("\n" + "="*60)
print("✅ FINAL MODEL TRAINING COMPLETE!")
print("="*60)