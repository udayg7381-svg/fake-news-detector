import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
import os
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 HEAVY DATASET MODEL TRAINING")
print("="*60)

# ============================================
# 1. LOAD DATASETS
# ============================================
def load_kaggle_dataset():
    """Load Kaggle Fake News Dataset"""
    try:
        fake = pd.read_csv('dataset/Fake.csv')
        true = pd.read_csv('dataset/True.csv')
        
        fake['label'] = 1
        true['label'] = 0
        
        # Use 'text' column if exists, else use 'title' + 'text'
        if 'text' in fake.columns:
            fake['content'] = fake['text']
            true['content'] = true['text']
        elif 'title' in fake.columns and 'text' in fake.columns:
            fake['content'] = fake['title'] + ' ' + fake['text']
            true['content'] = true['title'] + ' ' + true['text']
        else:
            # Use first text column
            text_col = fake.columns[0]
            fake['content'] = fake[text_col]
            true['content'] = true[text_col]
        
        df = pd.concat([fake, true], ignore_index=True)
        print(f"✅ Kaggle Dataset loaded: {len(df)} rows")
        return df
    except Exception as e:
        print(f"❌ Kaggle load error: {e}")
        return None

def load_liar_dataset():
    """Load LIAR Dataset (TSV)"""
    try:
        if not os.path.exists('dataset/train.tsv'):
            return None
        df = pd.read_csv('dataset/train.tsv', sep='\t', header=None)
        df.columns = ['id', 'statement', 'label', 'subject', 'speaker', 
                      'job', 'state', 'party', 'barely_true', 'false', 
                      'half_true', 'mostly_true', 'pants_fire', 'context']
        
        label_map = {
            'true': 0, 'mostly-true': 0, 'half-true': 0,
            'false': 1, 'mostly-false': 1, 'pants-fire': 1
        }
        df['label'] = df['label'].map(label_map)
        df = df.dropna(subset=['label'])
        df['content'] = df['statement']
        print(f"✅ LIAR Dataset loaded: {len(df)} rows")
        return df
    except:
        return None

# Load all datasets
print("\n📁 Loading datasets...")
datasets = []

kaggle_df = load_kaggle_dataset()
if kaggle_df is not None:
    datasets.append(kaggle_df)

liar_df = load_liar_dataset()
if liar_df is not None:
    datasets.append(liar_df)

if not datasets:
    print("❌ No datasets found!")
    print("📥 Please download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    exit()

# Merge all datasets
df = pd.concat(datasets, ignore_index=True)
print(f"\n✅ Combined dataset: {len(df)} rows")
print(f"Real (0): {len(df[df['label']==0])}")
print(f"Fake (1): {len(df[df['label']==1])}")

# ============================================
# 2. CLEAN TEXT
# ============================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\n🧹 Cleaning text...")
df['clean_text'] = df['content'].apply(clean_text)
df = df[df['clean_text'].str.len() > 20]  # Keep only meaningful text
print(f"✅ After cleaning: {len(df)} rows")

if len(df) == 0:
    print("❌ No data after cleaning! Please check dataset.")
    print("📊 Columns in dataset:", df.columns.tolist())
    exit()

# ============================================
# 3. BALANCE CLASSES
# ============================================
print("\n⚖️ Balancing classes...")
real = df[df['label']==0]
fake = df[df['label']==1]
min_count = min(len(real), len(fake))
print(f"Real: {len(real)}, Fake: {len(fake)}")
df_balanced = pd.concat([real.sample(min_count, random_state=42), 
                         fake.sample(min_count, random_state=42)]).sample(frac=1, random_state=42)
print(f"✅ Balanced: {len(df_balanced)} rows")

# ============================================
# 4. FEATURE ENGINEERING
# ============================================
print("\n🔄 TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)
X = vectorizer.fit_transform(df_balanced['clean_text'])
y = df_balanced['label']
print(f"✅ TF-IDF shape: {X.shape}")

# ============================================
# 5. TRAIN TEST SPLIT
# ============================================
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Training: {X_train.shape[0]} samples")
print(f"✅ Test: {X_test.shape[0]} samples")

# ============================================
# 6. SMOTE (Balance training data)
# ============================================
print("\n⚖️ Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"✅ After SMOTE: {X_train_balanced.shape[0]} samples")

# ============================================
# 7. TRAIN MODELS
# ============================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='linear', random_state=42, probability=True)
}

results = []
print("\n🚀 Training models...")

for name, model in models.items():
    print(f"\n📊 Training {name}...")
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    print(f"   ✅ Accuracy: {accuracy:.4f}")
    print(f"   ✅ Precision: {precision:.4f}")
    print(f"   ✅ Recall: {recall:.4f}")
    print(f"   ✅ F1-Score: {f1:.4f}")

# ============================================
# 8. SAVE BEST MODEL
# ============================================
print("\n💾 Saving best model...")
results_df = pd.DataFrame(results)
best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
best_model = models[best_model_name]

joblib.dump(best_model, 'models/heavy_model.pkl')
joblib.dump(vectorizer, 'models/heavy_vectorizer.pkl')
print(f"🏆 Best Model: {best_model_name}")
print("✅ Saved as 'models/heavy_model.pkl'")

print("\n" + "="*60)
print("📊 MODEL COMPARISON")
print("="*60)
print(results_df.to_string(index=False))

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)