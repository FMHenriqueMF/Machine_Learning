## Improve from collect_data_from_csv.py using the train_test_split module and sync the results proportionally 

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 20

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
dados = pd.read_csv(uri)


x = dados[["home","how_it_works","contact"]]
y = dados["bought"]

train_x, test_x, train_y, test_y = train_test_split(x, y,
                                                    random_state = SEED,
                                                    test_size = 0.5,
                                                    stratify = y)


print("Treinaremos com %d elementos e testaremos com %d elementos" %(len(train_x),len(test_x)))


model = LinearSVC()
model.fit(train_x,train_y)
pred = model.predict(test_x)

acc = accuracy_score(test_y, pred)
print(acc)
print(train_y.value_counts())
print(test_y.value_counts())
