import sklearn.naive_bayes as nb
import sklearn.neighbors as knn
import re
import random

price_list = []
fea_all_list = []

price_list_test = []
fea_all_list_test = []
rate = 0.95

with open('housing.data.txt', 'r') as  read:
    for line in read:
        data = re.split('\s+', line.strip())
        length = len(data)
        fea_list = []
        fea_list_test = []
        n = random.random()
        for idx in range(length):
            if idx + 1 == length:
                if n <= rate:
                    price_list.append(int(float(data[idx]) * 100))
                else:
                    price_list_test.append(int(float(data[idx]) * 100))
            else:
                if n <= rate:
                    fea_list.append(float(data[idx]))
                else:
                    fea_list_test.append(float(data[idx]))

        if len(fea_list) > 0:
            fea_all_list.append(fea_list)
        if len(fea_list_test) > 0:
            fea_all_list_test.append(fea_list_test)

model = knn.KNeighborsRegressor()

model.fit(fea_all_list, price_list)

rs_y = model.predict(fea_all_list_test)

for y in range(len(rs_y)):
    print(str(rs_y[y]) + '\t' + str(price_list_test[y]))

import numpy as np

np_rs_y = np.array(rs_y)
np_test_y = np.array(price_list_test)
y = (np_rs_y - np_test_y)
print(y)


# import joblib
#
# joblib.dump(model, 'model.data')
#
# print('=' * 100)
# model_load = joblib.load('model.data')
# rs_y = model_load.predict(fea_all_list_test)
#
# for y in range(len(rs_y)):
#     print(str(rs_y[y]) + '\t' + str(price_list_test[y]))
