import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import warnings

warnings.simplefilter("ignore")

data = [[i for i in i.strip().split(",")] for i in open("assets/abalone.data").readlines()]
X = [i[1:] for i in data]
y = [i[0] for i in data]

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)


class GridSearchHelper:

    def __init__(self, models_, params_):
        self.models = models_
        self.params = params_
        self.keys = models.keys()
        self.searches = {}

    def run(self, X, y):
        for key in self.keys:
            print("Running", key)
            model = self.models[key]
            params = self.params[key]
            print(model.get_params())
            search = GridSearchCV(model, params, cv=5)
            search.fit(X, y)
            self.searches[key] = search

    def score(self):
        frames = []
        for name, search in self.searches.items():
            frame = pd.DataFrame(search.cv_results_)
            frame = frame.filter(regex='^(?!.*param_).*$')
            frame['estimator'] = len(frame)*[name]
            frames.append(frame)
        df = pd.concat(frames)

        df = df.sort_values(['mean_test_score'], ascending=False)
        df = df.reset_index()
        df = df.drop(['rank_test_score', 'index'], 1)

        columns = df.columns.tolist()
        columns.remove('estimator')
        columns = ['estimator'] + columns
        df = df[columns]
        return df

    def best_score(self):
        for key in self.keys:
            print('Best params for {} is {}'.format(key, self.searches[key].best_params_))


models = {"Linear": SVC(kernel="linear"),
          "RBF": SVC(kernel="rbf"),
          "Sigmoid": SVC(kernel="sigmoid"),
          }


parameters = {
    'Linear': {'C': [2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7],
               # 'gamma': [2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4]},
               'gamma': [0.125]},
    'RBF': {'C': [2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7],
            # 'gamma': [2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4]},
            'gamma': [0.25]},
    'Sigmoid': {'C': [2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7],
                # 'gamma': [2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4]
                'gamma': [0.125]}
}

helper = GridSearchHelper(models, parameters)
helper.run(X_train, y_train)
print(helper.score())
print(helper.best_score())
