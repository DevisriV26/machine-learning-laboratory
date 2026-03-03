# ============================================
# STEP 1: Install Libraries
# ============================================
!pip install scikit-learn gradio joblib

# ============================================
# STEP 2: Import Libraries
# ============================================
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_text

import joblib
import gradio as gr

# ============================================
# STEP 3: Load Default Dataset (Colab Available)
# ============================================
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)

print("Dataset Shape:", df.shape)

# ============================================
# STEP 4: Convert Dataset → Student Performance
# ============================================

df = df.rename(columns={
    df.columns[0]: "study_hours",
    df.columns[1]: "attendance",
    df.columns[2]: "internal_marks",
    df.columns[3]: "assignment_score"
})

# Normalize values
df["study_hours"] = df["study_hours"] / df["study_hours"].max() * 10
df["attendance"] = df["attendance"] / df["attendance"].max() * 100
df["internal_marks"] = df["internal_marks"] / df["internal_marks"].max() * 100
df["assignment_score"] = df["assignment_score"] / df["assignment_score"].max() * 100

# Create PASS / FAIL label
df["result"] = (
    df["internal_marks"] +
    df["assignment_score"] +
    df["attendance"]
)/3

df["result"] = df["result"].apply(lambda x: 1 if x >= 50 else 0)

# ============================================
# STEP 5: Select Features
# ============================================
X = df[[
    "study_hours",
    "attendance",
    "internal_marks",
    "assignment_score"
]]

y = df["result"]

# ============================================
# STEP 6: Train-Test Split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# STEP 7: Train Random Forest Model
# ============================================
model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("\nModel Training Complete")

# ============================================
# STEP 8: Evaluation
# ============================================
y_pred = model.predict(X_test)

print("\nEvaluation Metrics")
print("===================")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================
# STEP 9: Show Different Decision Trees
# ============================================
print("\nShowing Random Forest Trees")

for i in range(3):
    print(f"\nTree {i+1}")
    tree_rules = export_text(
        model.estimators_[i],
        feature_names=list(X.columns)
    )
    print(tree_rules)

# ============================================
# STEP 10: Save Model
# ============================================
joblib.dump(model, "student_rf_model.pkl")

print("\nModel Saved Successfully")

# ============================================
# STEP 11: Gradio UI (Student Input)
# ============================================

model = joblib.load("student_rf_model.pkl")

def predict_performance(study, attend, internal, assignment):

    data = np.array([[study, attend, internal, assignment]])

    prediction = model.predict(data)[0]
    prob = model.predict_proba(data)[0]

    if prediction == 1:
        return f"PASS ✅ (Confidence {prob[1]:.2f})"
    else:
        return f"FAIL ❌ (Confidence {prob[0]:.2f})"

interface = gr.Interface(
    fn=predict_performance,
    inputs=[
        gr.Number(label="Study Hours"),
        gr.Number(label="Attendance (%)"),
        gr.Number(label="Internal Marks"),
        gr.Number(label="Assignment Score")
    ],
    outputs="text",
    title="Student Academic Performance Predictor (Random Forest)",
    description="Enter student details to predict PASS or FAIL"
)

interface.launch(share=True)
