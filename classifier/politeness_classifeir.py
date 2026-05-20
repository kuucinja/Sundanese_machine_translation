import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("dataset.csv")

# Features and labels
X = df["sundanese"]
y = df["politeness"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1,2),
        max_features=5000
    )),
    ("clf", LogisticRegression(
        max_iter=2000
    ))
])

# Train
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, preds))
print()
print(classification_report(y_test, preds))

# Save model
joblib.dump(model, "politeness_classifier.pkl")

print("Model saved.")

print("\nTrue label distribution (TEST SET):")
print(y_test.value_counts())