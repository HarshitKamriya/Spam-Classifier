import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.feature_extraction.text import CountVectorizer

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join back into string
    return ' '.join(tokens)


df = pd.read_csv("sms_spam.csv")
df.columns = ["label","message"]
df["label_num"] = df["label"].map({"ham":0,"spam":1})
df['clean_message'] = df['message'].apply(clean_text)

X_train,X_test,y_train,y_test = train_test_split(df["clean_message"],df["label_num"],test_size=0.2,random_state=42)


# Step 1: Fit on training data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Step 2: Transform test data using the same vectorizer
X_test_vec = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_vec,y_train)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test,y_pred)


dump(model,"spam_model.joblib")
dump(vectorizer,"vectorizer.joblib")
dump({"accuracy":accuracy},"model_metrics.joblib")