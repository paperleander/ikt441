#!/usr/bin/python3

import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import warnings

warnings.simplefilter("ignore")

data = [[i for i in i.strip().split(",")] for i in open("assets/abalone.data").readlines()]
X = [i[1:] for i in data]
y = [i[0] for i in data]

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

for i, j in zip(X_train, y_train): print(i, j)

# clf = SVC(kernel="linear", C=0.025, decision_function_shape='ovo')
# clf.fit(X_train, y_train)

names = ["Linear",
         "RBF",
         "Sigmoid",
         "SVC"]

classifiers = [SVC(kernel="linear", C=0.025),
               SVC(kernel="rbf", C=0.025),
               SVC(kernel="sigmoid", C=0.025),
               SVC(gamma=2, C=1)]

for i, cay in enumerate(classifiers):
    start = time.time()
    print("="*30)
    print(names[i])

    cay.fit(X_train, y_train)
    y_pred = cay.predict(X_test)

    total = X_test.shape[0]
    accuracy = (100 * ((y_test == y_pred).sum() / total))
    print("Accuracy = ", round(accuracy, 2), "%")
    print("Time: ", round(time.time() - start, 3), "s")
    print("="*30)
    print("")

