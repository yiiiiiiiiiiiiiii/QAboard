import pandas
import jieba.analyse
import sklearn.naive_bayes as nb
# import joblib
# import json
import random

max_len = 0

stop_word = set()
with open('stopwords.txt', 'r', encoding='utf-8') as read:
    for line in read.readlines():
        stop_word.add(line.strip())

feature_list = []
label_list = []

feature_test_list = []
label_test_list = []

word_dict = {}
word_list = []

rate = 0.9

pd = pandas.read_excel('chinesespam.xlsx')
rds = pd.to_records()
for rd in rds:
    nm = random.random()

    ft_list = []

    if rd[1].strip() == 'ham':
        if nm <= rate:
            label_list.append(1)
        else:
            label_test_list.append(1)
    else:
        if nm <= rate:
            label_list.append(0)
        else:
            label_test_list.append(0)

    wd_list = jieba.analyse.extract_tags(rd[2])
    for wd in wd_list:
        if wd in stop_word:
            continue

        if wd not in word_dict:
            word_list.append(wd)
            word_dict[wd] = len(word_list)

        ft_list.append(word_dict[wd])

    if nm <= rate:
        feature_list.append(ft_list)
    else:
        feature_test_list.append(ft_list)

    max_len = len(ft_list) if len(ft_list) > max_len else max_len

for p_list in feature_list:
    if len(p_list) != max_len:
        for i in range(max_len - len(p_list)):
            p_list.append(0)

for p_list in feature_test_list:
    if len(p_list) != max_len:
        for i in range(max_len - len(p_list)):
            p_list.append(0)

model_nb = nb.MultinomialNB()
model_nb.fit(feature_list, label_list)

y = model_nb.predict(feature_test_list)

y_len = len(y)
error_num = 0
for i in range(y_len):
    if y[i] != label_test_list[i]:
        error_num += 1
        print('real:' + str(label_test_list[i]) + ', predict:' + str(y[i]))

print('error count :' + str(error_num))
print('total count :' + str(y_len))
# print('error rate :' + str(error_num * 100.0 / y_len))

# joblib.dump(model_nb, 'model_nb')
# json.dump(word_list, open('word_list.json', 'w'))
# json.dump(word_dict, open('word_dict.json', 'w'))
