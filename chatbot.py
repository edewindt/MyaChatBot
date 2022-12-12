import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import keras.api._v2.keras as keras

from keras.models import load_model

lemmaitizer = WordNetLemmatizer
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('chatbot_model.model')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmaitizer.lemmatize(word) for word in sentence_words]
    return sentence

