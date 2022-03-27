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
    fisier = open("train.json")
    data = json.load(fisier)
    for i in data:
        for j in i['space_after']:
            space_after.append(j)
        for j in i['ner_tags']:
            ner_tags.append(j)
        for j in i['tokens']:
            tokens.append(j)
        for j in i['ner_ids']:
            ner_ids.append(str(j))
    # print(np.size(X))
    # print(np.size(Y))
    # print(tokens)
    # print(ner_ids)
    # print(np.size(tokens))
    split = int(len(tokens) * 0.95)
    training_sample = tokens[:split]
    training_label = ner_ids[:split]
    testing_sample = tokens[split:]
    testing_label = ner_ids[split:]
    tokenizer = Tokenizer(num_words=100000, oov_token="0", char_level=True)
    tokenizer.fit_on_texts(training_sample)
    word_index = tokenizer.word_index
    # print(np.size(word_index))
    training_sequences = tokenizer.texts_to_sequences(training_sample)
    training_sequences = pad_sequences(training_sequences, maxlen=32, padding='post', truncating='post')
    # print(type(training_sequences[0]))
    testing_sequences = tokenizer.texts_to_sequences(testing_sample)
    testing_sequences = pad_sequences(testing_sequences, maxlen=32, padding='post', truncating='post')
    # labelTokenizer = Tokenizer(num_words=100000, oov_token="0", char_level=True)
    # tokenizer.fit_on_texts(training_label)
    # word_index2 = labelTokenizer.word_index
    # training_labelS = labelTokenizer.texts_to_sequences(training_label)
    # training_label = pad_sequences(training_label, maxlen=16, padding='post', truncating='post')
    # testing_labelS = labelTokenizer.texts_to_sequences(testing_label)
    # testing_label = pad_sequences(testing_label, maxlen=16, padding='post', truncating='post')
    # print("sunt la model")
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(30000, 16, input_length=32),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(16, activation='linear')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(training_sequences)
    # print(training_label)
    # print("sunt la convertiri")
    training_sequences = np.array(training_sequences, dtype=np.float64)
    training_label = np.array(training_label, dtype=np.float64, like= training_sequences)
    testing_sequences = np.array(testing_sequences, dtype=np.float64)
    testing_label = np.array(testing_label, dtype=np.float64, like=testing_sequences)
    # training_label = np.asarray(training_label).astype('float64').reshape((300, 1))
    # testing_label = np.asarray(testing_label).astype('float64').reshape((300, 1))
    # print(training_label.shape)
    # print(training_sequences.shape)
    # print(testing_label.shape)
    # print(testing_sequences.shape)
    # print("sunt la history")
    history = model.fit(training_sequences, training_label, epochs=50,
                        validation_data=(testing_sequences, testing_label), verbose=2)

    fisierTest = open("test.json")
    dataTest = json.load(fisierTest)
    space_after2 = []
    tokens2 = []
    for i in dataTest:
        for j in i['space_after']:
            space_after2.append(j)
        for j in i['tokens']:
            tokens2.append(j)
    print(len(tokens2))
    testFinal_sequences = tokenizer.texts_to_sequences(tokens2)
    testFinal_sequences = pad_sequences(testFinal_sequences, maxlen=32, padding='post', truncating='post')
    output = model.predict(testFinal_sequences)
    output = output.tolist()
    finalList = []
    for i in output:
        row = []
        total = np.sum(i)
        row.append(str(len(finalList)))
        row.append(str((i/total).tolist().index(np.max(i/total))))
        finalList.append(row)
    with open('DragoonPredictions.csv', 'w', newline='') as outfile:
        write = csv.writer(outfile)
        write.writerow(['Id', 'ner_label'])
        write.writerows(finalList)
    # with open("sample.json", "w") as outfile:
    #     json.dump(y, outfile)
    # plot_graphs(history, "accuracy")
    # plot_graphs(history, "loss")
