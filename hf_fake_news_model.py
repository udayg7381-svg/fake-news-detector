from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import joblib

print("="*60)
print("🚀 LOADING SPECIALIZED FAKE NEWS MODEL")
print("="*60)

# Load model and tokenizer
model_name = "prajjwal1/bert-tiny-fake-news"
print(f"\n📁 Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create pipeline
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer
)
print("✅ Model loaded successfully!")

# Test function
def predict_fake_news(text):
    result = classifier(text)[0]
    label = result['label']  # 'LABEL_0' or 'LABEL_1'
    score = result['score'] * 100
    
    # Map: 'LABEL_0' = REAL, 'LABEL_1' = FAKE (based on model)
    if label == 'LABEL_0':
        return "REAL ✅", "real", score
    else:
        return "FAKE ❌", "fake", score

# Test with ISRO news
print("\n🧪 Testing with ISRO news...")
test_news = "ISRO successfully launched satellite today from Sriharikota. The satellite will help improve communication networks."
result, cls, conf = predict_fake_news(test_news)
print(f"Result: {result}")
print(f"Confidence: {conf:.2f}%")

# Save for fast loading
joblib.dump(classifier, 'models/hf_fake_news_model.pkl')
print("\n💾 Saved as 'models/hf_fake_news_model.pkl'")

print("\n" + "="*60)
print("✅ READY TO USE IN APP!")
print("="*60)