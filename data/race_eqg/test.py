import json
import nltk
import re

from pygments.unistring import No

with open("test.json") as file:
    data = json.load(file)

src = ''
for ele in data:
    src += ele['context'] + "[SEP]" + ele["answer"] + "\n"

with open('src_test.txt','w') as outfile:
    outfile.write(src)


# print(len(data))
# pattern = re.compile(u"[\u4e00-\u9fa5]+")
# res = ""
# preflag = False
# for ele in data:
#     tmp = ""
#     for c in ele["question"]:
#         # print(c, pattern.search(c))
#         if pattern.search(c) == None and preflag == True:
#             preflag = True
#             tmp += c
#         elif pattern.search(c) == None:
#             preflag = True
#             tmp += " " + c
#         elif c == "？":
#             tmp += " " + c
#         else:
#             preflag = False
#             tmp += " " + c
#     res += tmp.strip() + "\n"
#     # input(tmp.strip())
# with open("drcd_test_q.txt", "w") as file:
#     file.write(res.strip())


# for ele in data:
#     answer = ele["answer"]
#     if answer[-1] == ".":
#         answer = answer[:-2]
#     sp_answer = answer.split(" ")
#     for i in range(len(sp_answer), 0, -1):
#         for n in nltk.ngrams(answer.split(" "), i):
#             print(n)
#         input()


# json.dump(data[:100], open("train_keywords_100.json", "w"))
