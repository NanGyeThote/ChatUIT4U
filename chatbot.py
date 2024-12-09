import random
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model

# Initialize lemmatizer and load data
lemmatizer = nltk.WordNetLemmatizer()
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('uit_chat.h5')

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Create a bag of words for the given sentence."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predict the class of the given sentence using the model."""
    bow = bag_of_words(sentence)
    # print(f"Bag of words: {bow}")  # Debugging line
    res = model.predict(np.array([bow]))[0]
    # print(f"Model output: {res}")  # Debugging line
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

def get_response(intents_list, intents_json):
    """Get a response based on the predicted intent."""
    if not intents_list:
        return "Sorry, I didn't understand that."

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for intent in list_of_intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    
    return "Sorry, I didn't find an appropriate response."

# Console interface
print("GO! Bot is running!")

while True:
    message = input("You: ").strip()  # Trim input to handle extra spaces
    if message.lower() in ["exit", "quit", "bye"]:
        print("Bot: Goodbye!")
        break
    ints = predict_class(message)
    res = get_response(ints, intents)
    # print("Classes:", classes)  # This should print the list of intent tags
    print("Bot:", res)
