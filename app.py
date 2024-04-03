import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import re

ps = PorterStemmer()

# Load pre-trained model and vectorizer
best = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def process_text_data(text):
    # Your preprocessing steps here
    # ...
    return text

def test_pre(text):
    # Preprocess the input text
    text = process_text_data(text)
    text = text_process(text)
    text = text.split("\n")
    return text

def text_process(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

def find(p):
    if p == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess user input
    text = test_pre(input_sms)
    # Transform input using the vectorizer
    integers = vectorizer.transform(text)
    # Predict
    p = best.predict(integers)[0]
    # Display prediction
    find(p)
