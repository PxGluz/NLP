import json

if __name__ == "__main__":
    fisier = open("train.json")
    data = json.load(fisier)
    for i in data:
        for j in i['ner_tags']:
            print(j)
        for j in i['ner_ids']:
            print(j)
        for j in i['tokens']:
            print(j)
        for j in i['space_after']:
            print(j)
