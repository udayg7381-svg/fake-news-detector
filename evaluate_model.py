import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

print("="*60)
print("📊 MODEL EVALUATION")
print("="*60)

# Load model and data
print("\n📁 Loading model and data...")
model = joblib.load('models/best_model.pkl')
X_test = joblib.load('models/X_test.pkl')
y_test = joblib.load('models/y_test.pkl')
print("✅ Loaded successfully!")

# Predict
print("\n🔮 Making predictions...")
y_pred = model.predict(X_test)
print("✅ Predictions complete!")

# Confusion Matrix
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real (0)', 'Fake (1)'],
            yticklabels=['Real (0)', 'Fake (1)'])
plt.title('Confusion Matrix - Best Model (SVM)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('screenshots/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved as 'screenshots/confusion_matrix.png'")

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Real (0)', 'Fake (1)']))

# ROC Curve - Fixed Version
print("\n📈 Generating ROC Curve...")
try:
    # Try decision_function (SVM)
    y_scores = model.decision_function(X_test)
except AttributeError:
    # Fallback: use predict_proba (for other models)
    y_scores = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Best Model (SVM)')
plt.legend(loc="lower right")
plt.savefig('screenshots/roc_curve.png', dpi=300, bbox_inches='tight')
print(f"✅ ROC curve saved as 'screenshots/roc_curve.png' (AUC = {roc_auc:.4f})")

print("\n" + "="*60)
print("✅ MODEL EVALUATION COMPLETE!")
print("="*60)