import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import time
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 FAST MODEL TRAINING (No SVM)")
print("="*60)

# Load data
print("\n📁 Loading cleaned dataset...")
df = pd.read_csv('dataset/cleaned_dataset.csv')
X = df['clean_statement']
y = df['Label']
print(f"✅ Loaded {len(df)} rows")

# TF-IDF
print("\n🔄 Converting text to numbers...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X)
print(f"✅ TF-IDF shape: {X_tfidf.shape}")

# Train-Test Split
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Training: {X_train.shape[0]} samples")
print(f"✅ Test: {X_test.shape[0]} samples")

# SMOTE - Balance the classes!
print("\n⚖️ Applying SMOTE to balance classes...")
print(f"Before SMOTE - Real: {sum(y_train==0)}, Fake: {sum(y_train==1)}")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"After SMOTE - Real: {sum(y_train_balanced==0)}, Fake: {sum(y_train_balanced==1)}")

# Only Fast Models (SVM removed for speed)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []
print("\n🚀 Training fast models...")

for name, model in models.items():
    print(f"\n📊 Training {name}...")
    start_time = time.time()
    
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    train_time = time.time() - start_time
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Training Time': train_time
    })
    
    print(f"   ✅ Accuracy: {accuracy:.4f}")
    print(f"   ✅ Precision: {precision:.4f}")
    print(f"   ✅ Recall: {recall:.4f}")
    print(f"   ✅ F1-Score: {f1:.4f}")
    print(f"   ⏱️  Time: {train_time:.2f} seconds")

# Save best model
print("\n💾 Saving best model...")
results_df = pd.DataFrame(results)
best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
best_model = models[best_model_name]

joblib.dump(best_model, 'models/best_model_balanced.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer_balanced.pkl')
print(f"🏆 Best Model: {best_model_name}")
print("✅ Saved as 'models/best_model_balanced.pkl'")

print("\n" + "="*60)
print("📊 MODEL COMPARISON TABLE")
print("="*60)
print(results_df.to_string(index=False))

print("\n" + "="*60)
print("✅ FAST TRAINING COMPLETE!")
print("="*60)