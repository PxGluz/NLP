import json
import tensorflow as tf
import torch
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


if __name__ == "__main__":
    ner_tags = []
    ner_ids = []
    tokens = []
    space_after = []
    X = []
    Y = []
    spaces = []
    YNume = []
    fisier = open("train.json")
    data = json.load(fisier)
    for i in data:
        deIgnorat = []
        for j in i['space_after']:
            space_after.append(j)
        for j in i['ner_tags']:
            ner_tags.append(j)
        for j in i['tokens']:
            if ",.;()[]{}\\\"\'+=/-".find(j) == -1:
                tokens.append(j)
                deIgnorat.append(False)
            else:
                deIgnorat.append(True)
        k = 0
        for j in i['ner_ids']:
            if not deIgnorat[k]:
                ner_ids.append(j)
            k += 1
        X.append(tokens)
        Y.append(ner_ids)
        spaces.append(space_after)
        YNume.append(ner_tags)
        #
        ner_tags = []
        ner_ids = []
        tokens = []
        space_after = []
    #print(np.size(X))
    #print(np.size(Y))
    #print(X)
    #print(Y)
    split = int(len(X) * 0.85)
    training_sample = X[:split]
    training_label = Y[:split]
    testing_sample = X[split:]
    testing_label = Y[split:]
    tokenizer = Tokenizer(num_words=30000, oov_token="0")
    tokenizer.fit_on_texts(training_sample)
    word_index = tokenizer.word_index
    training_sequences = tokenizer.texts_to_sequences(training_sample)
    training_sequences = pad_sequences(training_sequences, maxlen=100, padding='post', truncating='post')
    testing_sequences = tokenizer.texts_to_sequences(testing_sample)
    testing_sequences = pad_sequences(testing_sequences, maxlen=100, padding='post', truncating='post')
    training_label = pad_sequences(training_label, maxlen=100, padding='post', truncating='post')
    testing_label = pad_sequences(testing_label, maxlen=100, padding='post', truncating='post')


    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(30000, 16, input_length=100),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    #print(training_sequences)
    print(training_label)
    print(len(training_sequences))
    print(len(training_label))
    training_sequences = torch.from_numpy(np.array(training_sequences, dtype=np.float64))
    training_label = torch.from_numpy(np.array(training_label, dtype=np.float64))
    testing_sequences = torch.from_numpy(np.array(testing_sequences, dtype=np.float64))
    testing_label = torch.from_numpy(np.array(testing_label, dtype=np.float64))
    history = model.fit(training_sequences, training_label, epochs=30, validation_data=(testing_sequences, testing_label), verbose=2)

    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")


