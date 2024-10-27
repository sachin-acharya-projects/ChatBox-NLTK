from tensorflow import keras

from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np

from typing import List, Dict
import random
import pickle
import json
import os
import pathlib
from colorama import Fore, init


init(True)

__all__ = ["clean_up_sentence", "bag_of_words", "predict_class", "get_response"]

intents_json = "./modelling/intents.json"
with open(intents_json) as file:
    intents = json.load(file)

MODEL_DIR = pathlib.Path("./model")

path_model = MODEL_DIR / "chatbox_model.keras"
classes_model = MODEL_DIR / "classes.pkl"
words_model = MODEL_DIR / "words.pkl"
lemmatizer = WordNetLemmatizer()

# For TensorFlow Log messages (reduce ammount of logging)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def clean_up_sentence(sentence: str):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]

    return sentence_words


def bag_of_words(sentence: str):
    with open(words_model, "rb") as file:
        words = pickle.load(file)
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence: str, e_threshold=0.25):
    with open(classes_model, "rb") as file:
        classes: List[str] = pickle.load(file)
    model: keras.Model = keras.models.load_model(path_model)

    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    results = [[i, r] for i, r in enumerate(res) if r > e_threshold]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list


def get_response(intents_list: List[Dict]):
    tag = intents_list[0]["intent"]
    list_of_intents = intents["intents"]

    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


while True:
    print(f"{Fore.LIGHTCYAN_EX}What do you want to ask?")
    message = input("> ")
    if message.lower() == "exit":
        exit()
    ints = predict_class(message)
    res = get_response(ints)

    print(f"{Fore.LIGHTGREEN_EX}{res}\n")
