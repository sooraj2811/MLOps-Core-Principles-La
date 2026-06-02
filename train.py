import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# 1. Classical Model (Model A): Using Support Vector Machine (SVM)
from sklearn.svm import SVC

# 2. Deep Learning framework (Model B): Using TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- PREPROCESSING ---
# Scale data before passing it to the Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- TRAIN MODEL A: Classical ML (SVM) ---
model_a = SVC(probability=True, random_state=42)
model_a.fit(X_train_scaled, y_train)

# Evaluate Model A
y_pred_a = model_a.predict(X_test_scaled)
acc_a = accuracy_score(y_test, y_pred_a)
f1_a = f1_score(y_test, y_pred_a)

# --- TRAIN MODEL B: Deep Learning (Feedforward NN) ---
# Updated to use tf.keras.Input to eliminate the user warning
model_b = Sequential([
    tf.keras.Input(shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Fixed typo: changed 'binary_cross_entropy' to 'binary_crossentropy'
model_b.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_b.fit(X_train_scaled, y_train, epochs=20, batch_size=16, verbose=0)

# Evaluate Model B
y_pred_b_probs = model_b.predict(X_test_scaled)
y_pred_b = (y_pred_b_probs > 0.5).astype(int).flatten()
acc_b = accuracy_score(y_test, y_pred_b)
f1_b = f1_score(y_test, y_pred_b)

print(f"Model A (Classical) - Accuracy: {acc_a:.4f}, F1: {f1_a:.4f}")
print(f"Model B (Deep Learning) - Accuracy: {acc_b:.4f}, F1: {f1_b:.4f}")

# --- MLOps Traceability & Artifact Retention ---
if acc_a >= acc_b:
    winning_model_type = "Classical"
    metrics = {"accuracy": acc_a, "f1_score": f1_a, "model_type": winning_model_type}
    
    # Save .pkl for Classical
    joblib.dump(model_a, 'best_model.pkl')
    print("Classical Model won and was saved to best_model.pkl")
else:
    winning_model_type = "Deep Learning"
    metrics = {"accuracy": acc_b, "f1_score": f1_b, "model_type": winning_model_type}
    
    # Save .h5 for Deep Learning
    model_b.save('best_model.h5')
    print("Deep Learning Model won and was saved to best_model.h5")

# Save winning model metrics to metrics.json
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
