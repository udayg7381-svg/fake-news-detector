from flask import Flask, request, render_template
import re
import string
import os
import requests
import joblib
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
import PyPDF2

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

print("="*60)
print("HEAVY ML MODEL (99.6% ACCURACY)")
print("="*60)

# ============================================
# LOAD HEAVY ML MODEL
# ============================================
print("[INFO] Loading Heavy ML model...")
try:
    model = joblib.load('models/heavy_model.pkl')
    vectorizer = joblib.load('models/heavy_vectorizer.pkl')
    print("[SUCCESS] Model loaded successfully!")
except FileNotFoundError:
    print("[ERROR] Model files not found! Please run train_model_heavy.py first.")
    print("[INFO] Falling back to rule-based detection...")
    model = None
    vectorizer = None
    USE_ML = False
else:
    USE_ML = True
print("="*60)

# ============================================
# CLEANING FUNCTION
# ============================================
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    text = ''
    try:
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except:
        return ''
    return text

# ============================================
# FALLBACK: RULE-BASED PATTERNS
# ============================================
FAKE_PATTERNS = [
    'miracle', 'cure', '100%', 'instant', 'magic', 'secret',
    'breakthrough', 'shocking', 'exposed', 'conspiracy',
    'free', 'guaranteed', 'works without', 'no battery',
    'without electricity', 'free energy', 'stop aging',
    'reverse aging', 'live forever', 'immortality',
    'replace sleep', 'never sleep', 'predict future',
    'made entirely of', 'turn green', 'aliens', 'ufo',
    'secret satellite', 'read messages', 'two suns'
]

REAL_PATTERNS = [
    'government announces', 'ministry of', 'parliament',
    'reserve bank', 'rbi', 'sebi', 'policy', 'regulation',
    'researchers develop', 'new study', 'research findings',
    'clinical trial', 'scientists', 'evidence shows',
    'records growth', 'renewable energy', 'gdp growth',
    'new railway', 'highway', 'metro', 'airport',
    'education program', 'university', 'new courses',
    'weather department', 'heavy rainfall', 'forecast'
]

# ============================================
# PREDICT FUNCTION WITH DEBUG
# ============================================
def predict_news(news_text):
    if not news_text:
        return None, None, None
    
    cleaned = clean_text(news_text)
    
    # =========================================
    # DEBUG: Print what's happening
    # =========================================
    print("\n" + "="*50)
    print("[DEBUG] PREDICTION DEBUG")
    print("="*50)
    print(f"[DEBUG] Original: {news_text[:80]}...")
    print(f"[DEBUG] Cleaned: {cleaned[:80]}...")
    
    # =========================================
    # ML MODEL PREDICTION (if available)
    # =========================================
    if USE_ML:
        try:
            text_vector = vectorizer.transform([cleaned])
            prediction = model.predict(text_vector)[0]
            proba = model.predict_proba(text_vector)[0]
            confidence = proba[prediction] * 100
            
            print(f"[DEBUG] ML Prediction: {prediction} (0=REAL, 1=FAKE)")
            print(f"[DEBUG] ML Probabilities: REAL={proba[0]:.4f}, FAKE={proba[1]:.4f}")
            print(f"[DEBUG] ML Confidence: {confidence:.2f}%")
            
            if prediction == 0:
                return "REAL ✅", "real", confidence
            else:
                return "FAKE ❌", "fake", confidence
                
        except Exception as e:
            print(f"[DEBUG] ML Error: {e}")
            print("[DEBUG] Falling back to rule-based...")
    
    # =========================================
    # RULE-BASED FALLBACK
    # =========================================
    fake_score = 0
    real_score = 0
    fake_matches = []
    real_matches = []
    
    for pattern in FAKE_PATTERNS:
        if pattern in cleaned:
            fake_score += 1
            fake_matches.append(pattern)
    
    for pattern in REAL_PATTERNS:
        if pattern in cleaned:
            real_score += 1
            real_matches.append(pattern)
    
    print(f"[DEBUG] Rule-Based Fake Score: {fake_score} (Matches: {fake_matches[:3]})")
    print(f"[DEBUG] Rule-Based Real Score: {real_score} (Matches: {real_matches[:3]})")
    
    if fake_score >= 2:
        confidence = min(95, 70 + (fake_score * 3))
        print(f"[DEBUG] Final Verdict: FAKE (Confidence: {confidence:.2f}%)")
        return "FAKE ❌", "fake", confidence
    elif real_score >= 3:
        confidence = min(95, 70 + (real_score * 3))
        print(f"[DEBUG] Final Verdict: REAL (Confidence: {confidence:.2f}%)")
        return "REAL ✅", "real", confidence
    elif fake_score >= 1:
        confidence = 65.0
        print(f"[DEBUG] Final Verdict: FAKE (Confidence: {confidence:.2f}%)")
        return "FAKE ❌", "fake", confidence
    elif real_score >= 1:
        confidence = 65.0
        print(f"[DEBUG] Final Verdict: REAL (Confidence: {confidence:.2f}%)")
        return "REAL ✅", "real", confidence
    else:
        confidence = 55.0
        print(f"[DEBUG] Final Verdict: UNCERTAIN (Confidence: {confidence:.2f}%)")
        return "UNCERTAIN ⚠️", "uncertain", confidence

# ============================================
# ROUTES
# ============================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        news_text = request.form.get('news_text', '')
        if not news_text:
            return render_template('index.html', error='Please enter some news text!')
        
        result, result_class, confidence = predict_news(news_text)
        
        return render_template('index.html',
                             news_text=news_text,
                             result=result,
                             confidence=f"{confidence:.2f}%",
                             result_class=result_class)
    
    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}')

@app.route('/predict_url', methods=['POST'])
def predict_url():
    try:
        url = request.form.get('url', '')
        if not url:
            return render_template('index.html', error='Please enter a URL!')
        
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        news_text = ' '.join([p.get_text() for p in paragraphs])
        
        if not news_text:
            return render_template('index.html', error='Could not extract news from URL!')
        
        result, result_class, confidence = predict_news(news_text)
        
        return render_template('index.html',
                             news_text=news_text[:500] + '...' if len(news_text) > 500 else news_text,
                             result=result,
                             confidence=f"{confidence:.2f}%",
                             result_class=result_class,
                             url=url)
    
    except Exception as e:
        return render_template('index.html', error=f'Error fetching URL: {str(e)}')

@app.route('/predict_file', methods=['POST'])
def predict_file():
    try:
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded!')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected!')
        
        if not allowed_file(file.filename):
            return render_template('index.html', error='Only .txt and .pdf files allowed!')
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if filename.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                news_text = f.read()
        elif filename.endswith('.pdf'):
            news_text = extract_text_from_pdf(filepath)
        else:
            os.remove(filepath)
            return render_template('index.html', error='Unsupported file format!')
        
        os.remove(filepath)
        
        if not news_text:
            return render_template('index.html', error='Could not extract text from file!')
        
        result, result_class, confidence = predict_news(news_text)
        
        return render_template('index.html',
                             news_text=news_text[:500] + '...' if len(news_text) > 500 else news_text,
                             result=result,
                             confidence=f"{confidence:.2f}%",
                             result_class=result_class)
    
    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}')

# ============================================
# RUN APP
# ============================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)