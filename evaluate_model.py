import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import os

# Import app to access model, vectorizer and preprocess function
import app

CSV_PATH = os.path.join(app.app.root_path, 'spam_ham_dataset.csv')
print('[eval] Using dataset:', CSV_PATH)

chunksize = 5000
y_true = []
y_proba = []

for chunk in pd.read_csv(CSV_PATH, usecols=['text','label_num'], chunksize=chunksize):
    texts = chunk['text'].fillna('').astype(str).tolist()
    labels = chunk['label_num'].astype(int).tolist()

    # Preprocess texts with same function used in app
    processed = [app.preprocess_email(t) for t in texts]
    X = app.vectorizer.transform(processed)

    probs = app.model.predict_proba(X)[:,1]

    y_true.extend(labels)
    y_proba.extend(probs.tolist())

# Convert to numpy
y_true = np.array(y_true)
y_proba = np.array(y_proba)

# Use same threshold
threshold = getattr(app, 'BEST_THRESHOLD', 0.5)

y_pred = (y_proba >= threshold).astype(int)

acc = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

# Compute AUC if both classes present
auc = None
try:
    auc = roc_auc_score(y_true, y_proba)
except Exception as e:
    print('[eval] AUC error:', e)

print('[eval] Samples:', len(y_true))
print('[eval] Threshold:', threshold)
print(f"[eval] Accuracy: {acc:.4f}")
print(f"[eval] Precision: {precision:.4f}")
print(f"[eval] Recall: {recall:.4f}")
print(f"[eval] F1: {f1:.4f}")
if auc is not None:
    print(f"[eval] AUC: {auc:.4f}")
else:
    print('[eval] AUC: could not compute')
