import random
import json
import pickle
import numpy as np
import nltk
import streamlit as st
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import os

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('wordnet')

# Paths for files
base_dir = os.path.dirname(__file__)
intents_path = os.path.join(base_dir, 'intents.json')
words_path = os.path.join(base_dir, 'words.pkl')
classes_path = os.path.join(base_dir, 'classes.pkl')
model_path = os.path.join(base_dir, 'uit_chat.h5')

# Load data and model
lemmatizer = WordNetLemmatizer()
intents = json.loads(open(intents_path).read())
words = pickle.load(open(words_path, 'rb'))
classes = pickle.load(open(classes_path, 'rb'))
model = load_model(model_path)

# Functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Streamlit UI
st.title("Your AI Companion for Mental Health and Stress Relief <3")
st.write("Hi there! How can I help you today?")

message = st.text_input("You: ", "")

try:
    if message:
        ints = predict_class(message)
        if ints:
            res = get_response(ints, intents)
            st.text_area("Bot:", value=res, height=100)
        else:
            st.text_area("Bot:", value="I'm sorry, I couldn't understand that. Can you rephrase?", height=100)
except Exception as e:
    st.error(f"An error occurred: {e}")
