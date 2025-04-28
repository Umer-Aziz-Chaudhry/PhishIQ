import os
import joblib
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Train the Model
# -----------------------------
def train_model():
    file_path = "Balanced_AI_Human.csv"
    logging.info(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)

    if 'sentence' not in df.columns or 'class' not in df.columns:
        raise ValueError("Dataset must contain 'sentence' and 'class' columns")

    X = df['sentence'].astype(str)
    y = df['class']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, "label_encoder.joblib")
    logging.info("LabelEncoder saved as label_encoder.joblib")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    logging.info("Training Logistic Regression model...")
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(C=1.0, solver='liblinear', max_iter=200))
    ])

    model.fit(X_train, y_train)
    joblib.dump(model, "ai_human_text_model.joblib")
    logging.info("Model saved as ai_human_text_model.joblib")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {acc:.4f}")

    target_names = [str(cls) for cls in label_encoder.classes_]

    print("\nClassification Report:\n", classification_report(
        y_test, y_pred, target_names=target_names
    ))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()
    logging.info("Confusion matrix saved as confusion_matrix.png")

# -----------------------------
# Prediction Function
# -----------------------------
def load_model():
    """Load the trained model."""
    model = joblib.load("ai_human_text_model.joblib")
    return model

def load_label_encoder():
    """Load the label encoder."""
    encoder = joblib.load("label_encoder.joblib")
    return encoder

label_mapping = {0: "Human", 1: "AI"}

def predict_text(full_text):
    """Predict if full text is AI or Human (without splitting)."""
    model = load_model()
    label_encoder = load_label_encoder()

    input_series = pd.Series([full_text])
    predictions = model.predict(input_series)
    decoded_predictions = [label_mapping[pred] for pred in predictions]

    return decoded_predictions[0]  # Returns "Human" or "AI"

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    # Train the model FIRST
    train_model()

    # Test Example
    test_text = [

        """Subject: Important: Scheduled System Maintenance Notification

Hello,

Please be advised that scheduled maintenance will occur on Sunday, April 30, 2025, from 2:00 AM to 6:00 AM UTC. During this time, certain services may be temporarily unavailable.

We appreciate your understanding and apologize for any inconvenience caused.

The Technical Operations Team


"""

    ]

    for idx, text in enumerate(test_text, 1):
        prediction = predict_text(text)
        print(f"\nSample {idx}:")
        print(f"Text: {text}")
        print(f"Predicted Source: {prediction}")
