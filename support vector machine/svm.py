# ============================================
# STEP 1: Install libraries
# ============================================
#!pip install datasets scikit-learn gradio joblib

# ============================================
# STEP 2: Import libraries
# ============================================
import pandas as pd
import numpy as np

from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import joblib
import gradio as gr

# ============================================
# STEP 3: Load dataset
# ============================================
dataset = load_dataset("mnemoraorg/spam-email-5k5")

df = dataset["train"].to_pandas()

print("Dataset shape:", df.shape)

# ============================================
# STEP 4: Preprocess data
# ============================================
df['label'] = df['Category'].map({
    'spam': 1,
    'ham': 0
})

X = df['Message']
y = df['label']

# ============================================
# STEP 5: Convert text to TF-IDF features
# ============================================
vectorizer = TfidfVectorizer(stop_words='english')

X_vectorized = vectorizer.fit_transform(X)

# ============================================
# STEP 6: Train-test split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================
# STEP 7: Train SVM model
# ============================================
model = SVC(kernel='linear', probability=True)

model.fit(X_train, y_train)

print("Model training complete")

# ============================================
# STEP 8: Evaluate model
# ============================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nEvaluation Metrics")
print("===================")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================
# STEP 9: Save model and vectorizer
# ============================================
joblib.dump(model, "svm_spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel saved successfully")

# ============================================
# STEP 10: Test with custom emails
# ============================================

# ============================================
# STEP 11: Gradio UI
# ============================================

# Load saved model
model = joblib.load("svm_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_spam(email):

    vec = vectorizer.transform([email])

    pred = model.predict(vec)[0]

    prob = model.predict_proba(vec)[0]

    spam_prob = prob[1]

    if pred == 1:
        return f"Spam (Confidence: {spam_prob:.2f})"
    else:
        return f"Not Spam (Confidence: {1-spam_prob:.2f})"

interface = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=5, placeholder="Enter email text here"),
    outputs="text",
    title="SVM Spam Email Detector",
    description="Enter an email message to check if it is spam"
)

interface.launch()
