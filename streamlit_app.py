import streamlit as st
import joblib

# Load your vectorizer and model
vect = joblib.load("tfidf_vectorizer.joblib")
clf  = joblib.load("logreg_model.joblib")

# Helper
def predict_sentiment(text: str) -> str:
    X = vect.transform([text])
    label = clf.predict(X)[0]
    return ["negative", "neutral", "positive"][label]

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("📝 Sentiment Analysis Demo")
st.write("Enter any product review or sentence, and I'll tell you if it's positive, neutral, or negative.")

user_input = st.text_area("Your text here", height=150)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        result = predict_sentiment(user_input)
        st.success(f"Sentiment: **{result.capitalize()}**")
