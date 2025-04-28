import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'AI_Text_Detection')))
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------
# Define TextCleaner
# --------------------------------------------
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.apply(self.clean_text)
    def clean_text(self, text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = ''.join(char for char in text if char.isalpha() or char.isspace())
        return text.strip()

# --------------------------------------------
# Paths
# --------------------------------------------
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SMS_MODEL_PATH = os.path.join(BASE_PATH, "SMS_Detection", "smishing_svm_model.joblib")
SMS_VECTORIZER_PATH = os.path.join(BASE_PATH, "SMS_Detection", "tfidf_vectorizer.joblib")
URL_MODEL_PATH = os.path.join(BASE_PATH, "URL_Detection", "url_detector_model.joblib")
EMAIL_MODEL_PATH = os.path.join(BASE_PATH, "Email_Detection", "email_detector.pkl")

# --------------------------------------------
# Load Models
# --------------------------------------------
sms_model = joblib.load(SMS_MODEL_PATH)
sms_vectorizer = joblib.load(SMS_VECTORIZER_PATH)
url_model = joblib.load(URL_MODEL_PATH)
email_model = joblib.load(EMAIL_MODEL_PATH)

# --------------------------------------------
# Detect Source (AI or Human)
# --------------------------------------------
def detect_source(text):
    """Predict if text is AI or Human by loading model and encoder inside."""
    try:
        model_path = os.path.join(BASE_PATH, "AI_Text_Detection", "ai_human_text_model.joblib")
        encoder_path = os.path.join(BASE_PATH, "AI_Text_Detection", "label_encoder.joblib")

        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)

        label_mapping = {0: "Human", 1: "AI"}

        input_series = pd.Series([text])
        predictions = model.predict(input_series)
        decoded_predictions = [label_mapping[pred] for pred in predictions]

        return decoded_predictions[0]
    except Exception as e:
        return f"Detection Error: {e}"

# --------------------------------------------
# Extract Features from URL
# --------------------------------------------
def extract_features_from_url(url):
    """Extract numerical features from a URL."""
    if not isinstance(url, str) or len(url.strip()) < 3:
        return None
    parsed = urlparse(url)
    return {
        'url_length': len(url),
        'num_dots': url.count('.'),
        'has_https': int('https' in url),
        'num_slashes': url.count('/'),
        'num_hyphens': url.count('-'),
        'num_underscores': url.count('_'),
        'num_parameters': url.count('&'),
        'has_ip': int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url))),
        'has_at_symbol': int('@' in url),
        'is_shortened': int(bool(re.search(r'bit\.ly|goo\.gl|tinyurl\.com|ow\.ly', url))),
        'hostname_length': len(parsed.hostname) if parsed.hostname else 0,
    }

# --------------------------------------------
# Flask App
# --------------------------------------------
app = Flask(__name__)
CORS(app)

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    input_text = data.get('input', '')
    detect_type = data.get('type', '').lower()

    if not input_text or detect_type not in ['email', 'sms', 'url']:
        return jsonify({'error': 'Invalid input or type'}), 400

    result = 'Unknown'
    source_label = 'N/A'

    if detect_type == 'url':
        # URL Detection
        features = extract_features_from_url(input_text)
        if not features:
            return jsonify({'error': 'Invalid or malformed URL'}), 400
        df = pd.DataFrame([features])
        prediction = url_model.predict(df)[0]
        result = 'Malicious' if prediction == 1 else 'Safe'
        source_label = None  # not applicable for URL

    elif detect_type == 'sms':
        # SMS Detection
        source_label = detect_source(input_text)

        # Smishing Detection
        cleaned = TextCleaner().transform(pd.Series([input_text]))
        tfidf_features = sms_vectorizer.transform(cleaned)

        def extract_sms_features(msg):
            msg = msg.lower()
            return {
                'message_length': len(msg),
                'word_count': len(msg.split()),
                'num_digits': len(re.findall(r'\d', msg)),
                'num_uppercase': sum(1 for c in msg if c.isupper()),
                'num_links': len(re.findall(r'http[s]?://\S+|www\.\S+', msg)),
                'has_call_to_action': int(bool(re.search(r'\b(buy|click|call|now|subscribe|free|win)\b', msg)))
            }

        sms_features = pd.DataFrame([extract_sms_features(input_text)]).fillna(0)
        from scipy.sparse import hstack
        combined = hstack([tfidf_features, sms_features.values])
        prediction = sms_model.predict(combined)[0]
        result = 'Smishing' if prediction == 1 else 'Safe'

    elif detect_type == 'email':
        # Email Detection
        source_label = detect_source(input_text)

        # Phishing Detection
        structured_features = {
            'email_length': len(input_text),
            'num_links': len(re.findall(r'http[s]?://', input_text)),
            'suspicious_keywords': len(re.findall(r'\b(account|verify|password|click|login|urgent)\b', input_text, re.I)),
            'uppercase_words': sum(1 for word in input_text.split() if word.isupper()),
            'special_characters': sum(1 for c in input_text if not c.isalnum() and c != ' '),
            'html_content': int(bool(re.search(r'<[^>]+>', input_text)))
        }

        full_input = pd.DataFrame([{
            'Email Text': input_text,
            **structured_features
        }])
        prediction = email_model.predict(full_input)[0]
        result = 'Phishing' if prediction == 1 else 'Safe'

    return jsonify({
        'result': result,
        'source': source_label
    })

if __name__ == '__main__':
    app.run(debug=True)
