import json

text = []
DATA = ['0', '1', '2','3','4']
for d in DATA:
    f = open('data/data_question_'+ d +'.json', 'r')
    data = json.load(f)
    N = len(data['true_message']['data'])

    for i in range(N):
       d = data['true_message']['data'][i]['text'] 
       text.append(d)


import ipdb
ipdb.set_trace()
print(text)





