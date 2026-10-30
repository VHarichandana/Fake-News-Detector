import streamlit as st
import pickle
import numpy as np
import nltk

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

#Data pre-processing    
import re
import string
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

#Wrap model loading in a function
@st.cache_resource
def load_model_and_vectorizer():
    with open("model/svm_fakeNews_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/fakeNews_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer()

st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection")
st.write("Paste a news article below to check whether it is **Fake** or **Real**.")

# Text input
news_text = st.text_area("Enter news text", height=200)

if st.button("Analyze"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize input
        cleaned_text = clean_text(news_text)
        if cleaned_text == "":
            st.warning("Text is too short after preprocessing. Please enter meaningful content.")
            st.stop()

        transformed_text = vectorizer.transform([cleaned_text])

        # Prediction
        prediction = model.predict(transformed_text)[0]

        # Confidence score using decision function
        decision_score = model.decision_function(transformed_text)[0]

        # Convert to confidence (0–100)
        confidence = 1 / (1 + np.exp(-abs(decision_score)))
        confidence = round(confidence * 100, 2)

        # Output
        if prediction == 1:
            st.success(f"✅ REAL NEWS\n\nConfidence: {confidence}%")
        else:
            st.error(f"🚨 FAKE NEWS\n\nConfidence: {confidence}%")
