"""
train_model.py
--------------
Trains a simple classification model on a synthetic student dataset and saves
the fitted pipeline artefact to model.pkl.

Run once before starting the FastAPI service:
    python train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# 1. Synthetic dataset
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
N = 1000

attendance        = np.random.randint(50, 100, N)          # % attendance
study_hours       = np.random.uniform(0.5, 8.0, N)         # hours/day
previous_scores   = np.random.randint(40, 100, N)          # previous exam %
assignments       = np.random.randint(0, 10, N)            # submitted out of 10
extracurricular   = np.random.randint(0, 2, N)             # 0/1 flag

# Simple deterministic rule used to create the label
score_index = (
    0.30 * attendance
    + 0.25 * study_hours * 10
    + 0.30 * previous_scores
    + 0.10 * assignments * 10
    + 0.05 * extracurricular * 100
)

result = np.where(score_index >= 65, "Pass", "Fail")

df = pd.DataFrame({
    "attendance":       attendance,
    "study_hours":      study_hours,
    "previous_scores":  previous_scores,
    "assignments":      assignments,
    "extracurricular":  extracurricular,
    "result":           result,
})

# ---------------------------------------------------------------------------
# 2. Split
# ---------------------------------------------------------------------------
FEATURES = ["attendance", "study_hours", "previous_scores", "assignments", "extracurricular"]
TARGET   = "result"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 3. Pipeline: scaler + classifier
# ---------------------------------------------------------------------------
pipeline = Pipeline([
    ("scaler",     StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
])

pipeline.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------
y_pred   = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------
# 5. Persist
# ---------------------------------------------------------------------------
MODEL_FILE = "model.pkl"
joblib.dump(pipeline, MODEL_FILE)
print(f"Model saved to: {os.path.abspath(MODEL_FILE)}")
