import random
import json
import pickle
import numpy as np
import nltk
import streamlit as st
from keras.models import load_model

# Initialize lemmatizer and load data
lemmatizer = nltk.WordNetLemmatizer()
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('uit_chat.h5')

# Function to clean and lemmatize the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create the bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

# Function to predict the class of the input sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

# Function to get a response based on the predicted intent
def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that."
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for intent in list_of_intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    
    return "Sorry, I didn't find an appropriate response."

# Streamlit App

st.title("UIT Chatbot")
st.write("Welcome to the University of Information Technology (UIT) Yangon chatbot!")

# Input box for user messages
user_input = st.text_input("You: ", "")

# Handle Exit phrases (exit, bye, goodbye, etc.)
exit_phrases = ["exit", "bye", "goodbye", "see you", "quit", "later"]
if user_input.lower() in exit_phrases:
    st.write("Bot: Goodbye! Have a great day.")
else:
    # Process user input and get response
    if user_input:
        ints = predict_class(user_input)
        response = get_response(ints, intents)
        st.write(f"Bot: {response}")
