import json
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer

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
        for j in i['space_after']:
            space_after.append(j)
        for j in i['ner_tags']:
            ner_tags.append(j)
        for j in i['ner_ids']:
            ner_ids.append(j)
        for j in i['tokens']:
            tokens.append(j)
        X.extend(tokens)
        Y.extend(ner_ids)
        spaces.extend(space_after)
        YNume.extend(ner_tags)
        #
        ner_tags = []
        ner_ids = []
        tokens = []
        space_after = []
    tokeniser = Tokenizer()
    split = int(len(X) * 0.85)
    training_sample = X[:split]
    training_label = Y[:split]
    testing_sample = X[split:]
    testing_label = Y[split:]

