import pandas as pd
from joblib import load
import streamlit as st



import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

model = load("spam_model.joblib")
vectorizer = load("vectorizer.joblib")
metric = load("model_metrics.joblib")



stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # remove URLs
    text = re.sub(r'\d+', '', text)                  # remove numbers
    text = re.sub(r'[^a-z\s]', '', text)             # remove punctuation
    tokens = nltk.word_tokenize(text)                # tokenize
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)





def check_spam(text):
    clean_txt = clean_text(text)
    text_vec = vectorizer.transform([clean_txt])
    prediction = model.predict(text_vec)[0]
    prob = model.predict_proba(text_vec)[0]
    return prediction, prob


st.title("Spam clasifier")

user_input = st.text_area("Enter a message to classify:")





st.sidebar.title("📊 Model Performance")
st.sidebar.write(f"Accuracy: **{metric['accuracy']*100:.2f}%**")



if st.button("Classify"):
    if user_input.strip() != "":
        prediction,prob = check_spam(user_input)

        spam_idx = list(model.classes_).index(1)  # find index of spam class
        st.write(f"Spam probability: {prob[spam_idx]*100:.2f}%")


        # Show result
        if prediction == 1:
            st.error("🚨 This message is Spam!")
        else:
            st.success("✅ This message is Not Spam.")
    else:
        st.warning("Please enter a message first.")
