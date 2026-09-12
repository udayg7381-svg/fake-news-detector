import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🤖 MODEL TRAINING")
print("="*60)

# Step 1: Load prepared data
print("\n📁 Loading prepared data...")
X_train = joblib.load('models/X_train.pkl')
X_test = joblib.load('models/X_test.pkl')
y_train = joblib.load('models/y_train.pkl')
y_test = joblib.load('models/y_test.pkl')
print(f"✅ Training data: {X_train.shape[0]} samples")
print(f"✅ Test data: {X_test.shape[0]} samples")

# Step 2: Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42)
}

# Step 3: Train and evaluate each model
results = []
print("\n🚀 Training models...")

for name, model in models.items():
    print(f"\n📊 Training {name}...")
    start_time = time.time()
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Training time
    train_time = time.time() - start_time
    
    # Store results
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

# Step 4: Save the best model
print("\n💾 Saving best model...")
results_df = pd.DataFrame(results)
best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
print(f"🏆 Best Model: {best_model_name}")

# Save best model
best_model = models[best_model_name]
joblib.dump(best_model, 'models/best_model.pkl')
print(f"✅ Best model saved as 'models/best_model.pkl'")

# Step 5: Display comparison table
print("\n" + "="*60)
print("📊 MODEL COMPARISON TABLE")
print("="*60)
print(results_df.to_string(index=False))

# Step 6: Save results to CSV
results_df.to_csv('models/model_comparison.csv', index=False)
print("\n✅ Results saved to 'models/model_comparison.csv'")

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)