import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🎯 EXPERT MODEL TRAINING - FORCED REAL NEWS LEARNING")
print("="*60)

# Load dataset
print("\n📁 Loading dataset...")
df = pd.read_csv('dataset/cleaned_dataset.csv')
print(f"✅ Loaded {len(df)} rows")

# REAL news examples to force learn
real_news_forced = [
    "ISRO successfully launched satellite from Sriharikota",
    "Chandrayaan mission successfully landed on moon",
    "ISRO scientist confirmed satellite is working properly",
    "India space agency launched communication satellite",
    "Government announced new education policy",
    "Supreme Court upheld constitutional validity",
    "Finance Ministry unveiled annual budget",
    "Health Ministry launched tuberculosis campaign",
    "Scientists discovered new species",
    "Researchers developed water purification system",
    "WHO approved indigenous vaccine",
    "IIT Bombay developed new technology",
    "GDP growth reached 7.8 percent",
    "RBI kept repo rate unchanged",
    "Manufacturing sector showing recovery",
    "Export growth increased significantly",
    "ISRO satellite launch was successful today",
    "India space research organisation launched mission",
    "Communication satellite launched by ISRO",
    "Space research organisation launched satellite"
]

# Create forced REAL news dataset
real_df = pd.DataFrame({
    'clean_statement': real_news_forced,
    'Label': [0] * len(real_news_forced)
})

# Add to original
df_combined = pd.concat([df, real_df], ignore_index=True)
print(f"✅ Added {len(real_news_forced)} forced REAL news examples")

# Balance classes - take minimum of both
real_count = len(df_combined[df_combined['Label']==0])
fake_count = len(df_combined[df_combined['Label']==1])
min_count = min(real_count, fake_count)

print(f"\n⚖️ Balancing classes... (taking {min_count} each)")

real_df_balanced = df_combined[df_combined['Label']==0].sample(n=min_count, random_state=42)
fake_df_balanced = df_combined[df_combined['Label']==1].sample(n=min_count, random_state=42)

df_final = pd.concat([real_df_balanced, fake_df_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\n✅ Final dataset: {len(df_final)} rows")
print(f"Real: {len(df_final[df_final['Label']==0])}")
print(f"Fake: {len(df_final[df_final['Label']==1])}")

# Save
df_final.to_csv('dataset/expert_balanced_dataset.csv', index=False)
print("💾 Saved as 'dataset/expert_balanced_dataset.csv'")

# TF-IDF
print("\n🔄 Converting text to numbers...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 3)
)
X = vectorizer.fit_transform(df_final['clean_statement'])
y = df_final['Label']
print(f"✅ TF-IDF shape: {X.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Training: {X_train.shape[0]} samples")
print(f"✅ Test: {X_test.shape[0]} samples")

# Train Random Forest
print("\n🌳 Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
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
print("\n💾 Saving expert model...")
joblib.dump(model, 'models/expert_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer_expert.pkl')
print("✅ Saved as 'models/expert_model.pkl'")

print("\n" + "="*60)
print("✅ EXPERT MODEL TRAINING COMPLETE!")
print("="*60)