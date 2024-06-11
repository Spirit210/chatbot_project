import csv
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load the intents file
intents = []
with open('data/intents.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        intents.append({
            'tag': row['tag'],
            'patterns': row['patterns'].split('|'),
            'responses': row['responses'].split('|')
        })

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Process each intent
for intent in intents:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add documents in the corpus
        documents.append((word_list, intent['tag']))
        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

print(f"Words: {words}")
print(f"Classes: {classes}")
print(f"Documents: {documents}")
