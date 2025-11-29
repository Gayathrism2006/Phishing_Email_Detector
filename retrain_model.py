import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.utils import shuffle
import joblib
from sklearn.feature_extraction import text
import os

print("[train] Starting model retraining...")

# Setup
CSV_PATH = 'spam_ham_dataset.csv'
stop_words = text.ENGLISH_STOP_WORDS

def preprocess_email(text_in):
    # Same preprocessing as app.py but without the legitimate_indicators boost
    text_in = text_in.lower()
    text_in = re.sub(r'[^a-zA-Z\s]', ' ', text_in)
    text_in = re.sub(r'\s+', ' ', text_in).strip()
    text_in = ' '.join([word for word in text_in.split() if word not in stop_words])
    return text_in

# Load full dataset
print("[train] Loading dataset...")
df = pd.read_csv(CSV_PATH, usecols=['text', 'label_num'])
df = df.dropna()
print(f"[train] Dataset size: {len(df)}")
print(f"[train] Class distribution:\n{df['label_num'].value_counts()}")

# Preprocess
df['text_clean'] = df['text'].apply(preprocess_email)

# Split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['text_clean'], df['label_num'],
    test_size=0.2, stratify=df['label_num'], random_state=42
)

print(f"[train] Train size: {len(X_train_text)}, Test size: {len(X_test_text)}")

# Vectorize
print("[train] Vectorizing...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Try multiple models and pick best
models = {
    'RandomForest': RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42, max_depth=20),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
}

best_model_name = None
best_model = None
best_f1 = 0

for name, model in models.items():
    print(f"\n[train] Training {name}...")
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

print(f"\n[train] Best model: {best_model_name} with F1: {best_f1:.4f}")

# Fine-tune threshold on best model
print("\n[train] Optimizing threshold...")
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

best_thresh = 0.5
best_f1_thresh = 0
for thresh in np.linspace(0, 1, 101):
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    f1_thresh = f1_score(y_test, y_pred_thresh)
    if f1_thresh > best_f1_thresh:
        best_f1_thresh = f1_thresh
        best_thresh = thresh

print(f"[train] Optimal threshold: {best_thresh:.2f} with F1: {best_f1_thresh:.4f}")

# Evaluate with optimized threshold
y_pred_final = (y_pred_proba >= best_thresh).astype(int)

print("\n" + "="*60)
print("FINAL EVALUATION (Test Set)")
print("="*60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_final):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_final):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_final):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_final):.4f}")
print(f"AUC:       {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"Threshold: {best_thresh:.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=['Ham', 'Unsafe']))

# Save
print("\n[train] Saving models...")
joblib.dump(best_model, 'binary_email_classifier.pkl')
joblib.dump(vectorizer, 'binary_email_tfidf_vectorizer.pkl')
print("[train] ✓ Models saved successfully!")
print("[train] New threshold should be: " + str(best_thresh))
