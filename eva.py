from sklearn.metrics import f1_score
import numpy as np

import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Download required NLTK resources
# nltk.download('punkt')
# nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents file
with open('./intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []

ignore_letters = ['?', '!', '.', ',']

# Process intents to extract words, classes, and documents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and filter words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Sort and save words and classes
classes = sorted(set(classes))
print(f"Number of classes: {len(classes)}")

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = [1 if word in [lemmatizer.lemmatize(w.lower()) for w in document[0]] else 0 for word in words]
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

# Shuffle and convert to numpy array
random.shuffle(training)
training = np.array(training)

train_x = training[:, :len(words)]
train_y = training[:, len(words):]

# Split data into training and validation sets
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Assuming you have a trained model and data like:
# val_x: Validation features
# val_y: True labels (one-hot encoded)

# Step 1: Make predictions on the validation set
model = tf.keras.models.load_model('uit_chat.h5')
val_predictions = model.predict(val_x)

# Step 2: Convert predictions to class labels
val_pred_labels = np.argmax(val_predictions, axis=1)  # Getting the index of the max value for each prediction

# Step 3: Convert true labels (val_y) from one-hot encoding to class labels
val_true_labels = np.argmax(val_y, axis=1)
# Accuracy
accuracy = accuracy_score(val_true_labels, val_pred_labels)

# Precision
precision = precision_score(val_true_labels, val_pred_labels, average='weighted')  # 'weighted' handles imbalanced data

# Recall
recall = recall_score(val_true_labels, val_pred_labels, average='weighted')

# F1 Score
f1 = f1_score(val_true_labels, val_pred_labels, average='weighted')

# Print all metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")