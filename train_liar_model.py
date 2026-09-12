import pandas as pd
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🎯 TRAINING WITH LIAR DATASET")
print("="*60)

# Load data
print("\n📁 Loading LIAR dataset...")
df = pd.read_csv('dataset/train.tsv', sep='\t', header=None)
df.columns = ['id', 'statement', 'label', 'subject', 'speaker', 
              'job', 'state', 'party', 'barely_true', 'false', 
              'half_true', 'mostly_true', 'pants_fire', 'context']

print(f"✅ Loaded {len(df)} rows")

# Convert labels
label_map = {
    'true': 0, 'mostly-true': 0, 'half-true': 0,
    'false': 1, 'mostly-false': 1, 'pants-fire': 1
}
df['label_binary'] = df['label'].map(label_map)
df = df.dropna(subset=['label_binary'])

print(f"✅ After filtering: {len(df)} rows")
print(f"Real (0): {len(df[df['label_binary']==0])}")
print(f"Fake (1): {len(df[df['label_binary']==1])}")

# Clean text
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_statement'] = df['statement'].apply(clean_text)

# Remove empty
df = df[df['clean_statement'].str.len() > 10]
print(f"✅ After cleaning: {len(df)} rows")

# Features
X = df['clean_statement']
y = df['label_binary']

print(f"\n📊 Final dataset: {len(X)} rows")
print(f"Real (0): {len(y[y==0])}")
print(f"Fake (1): {len(y[y==1])}")

# TF-IDF
print("\n🔄 TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)
print(f"✅ Shape: {X_tfidf.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# Train
print("\n🚀 Training...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"\n📊 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"📊 F1-Score: {f1_score(y_test, y_pred):.4f}")

# Save
print("\n💾 Saving model...")
joblib.dump(model, 'models/liar_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer_liar.pkl')
print("✅ Saved as 'models/liar_model.pkl'")

print("\n" + "="*60)
print("✅ LIAR MODEL TRAINING COMPLETE!")
print("="*60)