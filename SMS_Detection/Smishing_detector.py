import os
import re
import joblib
import logging
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', text.lower()).strip()

# -----------------------------
# Main Function
# -----------------------------
def main():
    # Load enhanced dataset with features
    dataset_path = os.path.join(os.path.dirname(__file__), "Balanced_SMS_Dataset.csv")
    logging.info("Loading dataset with features...")
    df = pd.read_csv(dataset_path)

    # Label normalization
    df['label'] = df['label'].str.lower().replace({'smish': 1, 'ham': 0}).astype(int)


    # Clean text for TF-IDF
    df['message'] = df['message'].astype(str).apply(clean_text)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_text = vectorizer.fit_transform(df['message'])

    # Extract numeric features
    numeric_features = df[['message_length', 'word_count', 'num_digits', 'num_uppercase', 'num_links', 'has_call_to_action']].fillna(0)

    # Combine features
    X = hstack([X_text, numeric_features.values])
    y = df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SVM Model
    svm_model = SVC(kernel='linear', probability=True)
    logging.info("Training SVM model...")
    svm_model.fit(X_train, y_train)

    # Save model and vectorizer
    joblib.dump(svm_model, "smishing_svm_model.joblib")
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
    logging.info("Model and vectorizer saved.")

    # Evaluate
    y_pred = svm_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['safe', 'phishing']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['safe', 'phishing'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Smishing Detection Confusion Matrix")
    plt.savefig("smishing_confusion_matrix.png")
    plt.show()

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
