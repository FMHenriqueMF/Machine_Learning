import pandas as pd
import numpy as np
import graphviz
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier


### --- difining the SEED and getting the data --- ###
SEED = 5
np.random.seed(SEED)
uri = "https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv"
data = pd.read_csv(uri)
change_sold = { 'no':0, 'yes':1}
data.sold = data.sold.map(change_sold)
actual_year = datetime.today().year
data['model_age'] = actual_year - data.model_year
data['km_per_year'] = data.mileage_per_year * 1.60934
data = data.drop(columns = ["Unnamed: 0","mileage_per_year","model_year"])

x = data[["price","model_age","km_per_year"]]
y = data["sold"]

### --- training the data --- ###

train_x, test_x, train_y, test_y = train_test_split(x, y,
                                                    random_state = SEED,
                                                    test_size = 0.25,
                                                    stratify = y)


print("Treinaremos com %d elementos e testaremos com %d elementos" %(len(train_x),len(test_x)))

### --- creating a decision tree  --- ###
model = DecisionTreeClassifier(max_depth=5)
model.fit(train_x,train_y)
pred = model.score(test_x,test_y)
print(pred)

### --- Creating a Dummy classifier to serve as parameter --- ###
dummy_stratified = DummyClassifier()
dummy_stratified.fit(train_x,train_y)
pred_dummy1 = dummy_stratified.score(test_x,test_y)
print(pred_dummy1)

### --- creating a image of the decision tree --- ###
features = x.columns
dot_data = export_graphviz(model, out_file=None,filled = True,rounded = True, feature_names = features,class_names = ["no","yes"])
graph = graphviz.Source(dot_data)
graph.view()
