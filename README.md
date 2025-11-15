# 📧 Spam Classifier with Streamlit

Live Demo : https://spam-classifier-torgkx83xpvbzndfelztfh.streamlit.app/

A machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using **Naive Bayes** and a **Bag-of-Words model**.  
The app is built with **Streamlit** for an interactive GUI and supports user input, probability scores, and model performance metrics.

---

## 🚀 Features
- Train a **Multinomial Naive Bayes** classifier on SMS spam dataset.
- Preprocess text using **regex + NLTK** (cleaning, stopword removal, lemmatization).
- Interactive **Streamlit GUI**:
  - Enter a message and classify it as Spam/Ham.
  - Display **spam probability score**.
  - Show **model accuracy** in the sidebar.
  - Visualize confusion matrix and classification report.
- Save and load models using **Joblib** for efficiency.
- Support for retraining with user-provided datasets.

---

## 📂 Project Structure

├── app.py                # Streamlit app ├── train_model.py        # Training script ├── sms_spam.csv          # Dataset (SMS Spam Collection) ├── spam_model.joblib     # Saved trained model ├── vectorizer.joblib     # Saved CountVectorizer ├── model_metrics.joblib  # Saved accuracy score └── README.md             # Project documentation
