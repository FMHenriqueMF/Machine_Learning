import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

### --- defining seed and getting data --- ###
SEED = 5
np.random.seed(SEED)

uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
data = pd.read_csv(uri)

change = { 0:1,1:0}
data['finished'] = data.unfinished.map(change)

### --- Training the data --- ###
x = data[['expected_hours','price']]
y = data['finished']

raw_train_x, raw_test_x, train_y, test_y = train_test_split(x,y,
                                                            test_size = 0.25, 
                                                            stratify = y)

### --- changing graph scale --- ###

scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x)
test_x = scaler.transform(raw_test_x)

### --- training and predicting the data --- ###
model = SVC()
model.fit(train_x,train_y)
pred = model.predict(test_x)

### --- getting the accuracy --- ###
acc = accuracy_score(test_y,pred) *100
print(acc)
base_pred = np.ones(540)
baseline_acc = accuracy_score(test_y,base_pred) *100
print(baseline_acc)

### --- rescalling the graph --- ###
data_x = test_x[:,0]
data_y = test_x[:,1]

x_min = data_x.min()
x_max = data_x.max()
y_min = data_y.min()
y_max = data_y.max()

pixels = 100
x_axis = np.arange(x_min,x_max, (x_max - x_min)/pixels)
y_axis = np.arange(y_min,y_max, (y_max - y_min)/pixels)

xx,yy = np.meshgrid(x_axis,y_axis)
dots = np.c_[xx.ravel(),yy.ravel()]

z = model.predict(dots)
z = z.reshape(xx.shape)

plt.contourf(xx,yy,z,alpha =0.4)
plt.scatter(data_x,data_y, c=test_y,s=1)
plt.show()
