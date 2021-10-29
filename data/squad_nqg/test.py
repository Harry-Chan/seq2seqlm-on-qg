import json
import nltk

with open("train_keywords.json") as file:
    data = json.load(file)

print(len(data))
# for ele in data:
# answer = ele["answer"]
# if answer[-1] == ".":
#     answer = answer[:-2]
# sp_answer = answer.split(" ")
# for i in range(len(sp_answer), 0, -1):
#     for n in nltk.ngrams(answer.split(" "), i):
#         print(n)
#     input()


# json.dump(data[:100], open("train_keywords_100.json", "w"))
