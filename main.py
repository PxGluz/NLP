import json
import csv
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
    # print(np.size(X))
    # print(np.size(Y))
    # print(X)
    # print(Y)
    split = int(len(X) * 0.95)
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
    training_label = pad_sequences(training_label, maxlen=16, padding='post', truncating='post')
    testing_label = pad_sequences(testing_label, maxlen=16, padding='post', truncating='post')

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(30000, 16, input_length=100),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(16, activation='linear')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(training_sequences)
    # print(training_label)
    # print(len(training_sequences))
    # print(len(training_label))
    training_sequences = np.array(training_sequences, dtype=np.float64)
    training_label = np.array(training_label, dtype=np.float64)
    testing_sequences = np.array(testing_sequences, dtype=np.float64)
    testing_label = np.array(testing_label, dtype=np.float64)
    # training_label = np.asarray(training_label).astype('float64').reshape((300, 1))
    # testing_label = np.asarray(testing_label).astype('float64').reshape((300, 1))
    # print(training_label.shape)
    # print(training_sequences.shape)
    # print(testing_label.shape)
    # print(testing_sequences.shape)
    history = model.fit(training_sequences, training_label, epochs=10,
                        validation_data=(testing_sequences, testing_label), verbose=2)

    fisierTest = open("test.json")
    dataTest = json.load(fisierTest)
    for i in dataTest:
        for j in i['space_after']:
            space_after.append(j)
        for j in i['tokens']:
            if ",.;()[]{}\\\"\'+=/-".find(j) == -1:
                tokens.append(j)
        X.append(tokens)
        spaces.append(space_after)
        #
        tokens = []
        space_after = []
    testFinal_sequences = tokenizer.texts_to_sequences(X)
    testFinal_sequences = pad_sequences(testFinal_sequences, maxlen=100, padding='post', truncating='post')
    output = model.predict(testFinal_sequences)
    output = output.tolist()
    finalList = []
    for i in output:
        row = []
        total = np.sum(i)
        row.append(str(len(finalList)))
        row.append(str((i/total).tolist().index(np.max(i/total))))
        finalList.append(row)
    with open('myfile.csv', 'w', newline='') as outfile:
        write = csv.writer(outfile)
        write.writerow(['Id', 'ner_label'])
        write.writerows(finalList)
    # with open("sample.json", "w") as outfile:
    #     json.dump(y, outfile)
    # plot_graphs(history, "accuracy")
    # plot_graphs(history, "loss")
