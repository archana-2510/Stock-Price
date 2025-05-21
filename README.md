# Stock-Price

# ai_stock_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import streamlit as st

# Load dataset
df = pd.read_csv("ai_stock_prediction_dataset.csv")

# Preprocess
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

# Feature columns
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment_Score', 'RSI', 'MACD']
X = df[features]
y = df['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# Streamlit Deployment UI
def run_app():
    st.title("AI Stock Trend Predictor")
    st.write("Upload stock indicators to predict trend (Up or Down).")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        user_data = pd.read_csv(uploaded)
        if set(features).issubset(user_data.columns):
            preds = model.predict(user_data[features])
            user_data['Prediction'] = ['Up' if p == 1 else 'Down' for p in preds]
            st.write(user_data)
        else:
            st.error("CSV must contain: " + ", ".join(features))

if __name__ == "__main__":
    run_app()
