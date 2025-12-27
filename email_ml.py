import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

# Load Dataset
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate the Model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

# Function to Predict Spam or Ham
def predict_message():
    message = entry.get()
    if message.strip() == "":
        messagebox.showwarning("Input Error", "Please enter a message.")
        return
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]
    result_label.config(text=f"Prediction: {'Spam' if prediction == 1 else 'Ham'}", fg="red" if prediction == 1 else "green")

# GUI Setup
root = tk.Tk()
root.title("Spam Classifier")
root.geometry("500x400")
root.config(bg="#e3f2fd")

frame = tk.Frame(root, bg="#90caf9", padx=20, pady=20, relief=tk.RIDGE, bd=5)
frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

title_label = tk.Label(frame, text="Spam Detector", font=("Arial", 18, "bold"), bg="#90caf9", fg="white")
title_label.pack(pady=10)

accuracy_label = tk.Label(frame, text=f"Model Accuracy: {accuracy:.2%}", font=("Arial", 12, "bold"), bg="#90caf9", fg="white")
accuracy_label.pack()

entry = ttk.Entry(frame, width=50, font=("Arial", 12))
entry.pack(pady=15)

check_button = ttk.Button(frame, text="Check Message", command=predict_message)
check_button.pack()

result_label = tk.Label(frame, text="", font=("Arial", 14, "bold"), bg="#90caf9")
result_label.pack(pady=15)

quit_button = ttk.Button(frame, text="Quit", command=root.quit)
quit_button.pack(pady=10)

root.mainloop()
