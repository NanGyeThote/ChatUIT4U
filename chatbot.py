import streamlit as st
import nltk
from nltk.tokenize import word_tokenize

# Ensure the punkt tokenizer is downloaded at runtime
nltk.download('punkt')

st.title("Tokenization Example")

# Text input for user to provide text
text = st.text_input("Enter a sentence to tokenize:")

if text:
    try:
        # Tokenize the input text
        tokens = word_tokenize(text)
        st.write("Tokens:", tokens)
    except Exception as e:
        st.error(f"An error occurred: {e}")
