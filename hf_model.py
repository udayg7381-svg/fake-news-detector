from transformers import pipeline
import joblib

print("="*60)
print("🤗 USING HUGGING FACE PRE-TRAINED MODEL")
print("="*60)

# Load pre-trained model for text classification
print("\n📁 Loading Hugging Face model...")
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("✅ Model loaded!")

# Test function
def predict_news_hf(news_text):
    result = classifier(news_text)[0]
    label = result['label']
    confidence = result['score'] * 100
    
    # Map to REAL/FAKE
    if label == 'POSITIVE':
        return "REAL ✅", "real", confidence
    else:
        return "FAKE ❌", "fake", confidence

# Save the pipeline for faster loading
joblib.dump(classifier, 'models/hf_model.pkl')
print("✅ Saved as 'models/hf_model.pkl'")

# Test with ISRO news
print("\n🧪 Testing with ISRO news...")
test_news = "ISRO successfully launched satellite today from Sriharikota. The satellite will help improve communication networks."
result, cls, conf = predict_news_hf(test_news)
print(f"Result: {result}")
print(f"Confidence: {conf:.2f}%")

print("\n" + "="*60)
print("✅ Ready to use in app!")
print("="*60)