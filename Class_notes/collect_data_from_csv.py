## Import CSV data and predict results

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
dados = pd.read_csv(uri)


x = dados[["home","how_it_works","contact"]]
y = dados["bought"]

train_x = x[:50]
train_y = y[:50]

test_x = x[50:]
test_y = y[50:]

print("We will training %d elements and will test %d elements" %(len(train_x),len(test_x)))


model = LinearSVC()
model.fit(train_x,train_y)
pred = model.predict(test_x)

acc = accuracy_score(test_y, pred)
print(acc)
