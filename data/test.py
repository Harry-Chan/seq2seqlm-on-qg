import json

with open('squad_nqg/train_keywords.json') as file:
    data = json.load(file)

for ele in data:
    del ele['noun_keywords']

json.dump(data,open('squad_nqg/train_keywords.json','w'))
