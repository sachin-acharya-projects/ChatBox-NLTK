from tensorflow import keras

from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np

from typing import List, Tuple
import random
import pickle
import json
import os
import pathlib


# For TensorFlow Log messages (reduce ammount of logging)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODEL_DIR = pathlib.Path("../model")
os.makedirs(MODEL_DIR, exist_ok=True)

with open("intents.json") as file:
    intents = json.load(file)

lemmantizer = WordNetLemmatizer()
words: List[str] = []
classes: List[str] = []
documents: Tuple[List[str], str] = []
ignore_symbols = ["?", "!", ".", ","]


for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))

        if intent["tag"] not in classes:
            classes.append(intent["tag"])


words = [lemmantizer.lemmatize(word) for word in words if word not in ignore_symbols]
words = sorted(set(words))
classes = sorted(set(classes))

with open(MODEL_DIR / "words.pkl", "wb") as file:
    pickle.dump(words, file)
with open(MODEL_DIR / "classes.pkl", "wb") as file:
    pickle.dump(classes, file)


training = []
output_empty = [0] * len(classes)


for document in documents:
    bag: List[int] = []
    word_patterns = document[0]
    word_patterns = [lemmantizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1 if word in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype="object")
train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = keras.models.Sequential()
model.add(keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(train_y[0]), activation="softmax"))

sgd = keras.optimizers.SGD(
    learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True
)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=5, verbose=1)
model.save(MODEL_DIR / "chatbox_model.keras")
