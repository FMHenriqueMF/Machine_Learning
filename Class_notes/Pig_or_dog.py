##Basic machine learn #1
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


### --- Define the training --- ###
# 1st argument: Has long fur? 
# 2nd argument: Has short paw?
# 3rd argument: Does it bark?
# If the answear of any question is YES, receive 1. Otherwise, receive 0.

pig1 = [0,1,0]   
pig2 = [0,1,1]
pig3 = [1,1,0]

dog1 = [0,1,1]
dog2 = [1,0,1]
dog3 = [1,1,1]

train_x = [pig1,pig2,pig3,dog1,dog2,dog3]
train_y = [1,1,1,0,0,0]       #If the Y receive 1, it is a pig. If receive 0, it is a dog


model = LinearSVC()
model.fit(train_x,train_y)

## after training, lets give some mystery animal to let the program say if it is a dog or a pig
mysteryAnimal = [1,1,1]

mystery1 = [1,1,1]  #long fur, short paw, barks
mystery2 = [1,1,0]  #long fur, short paw
mystery3 = [0,1,1]  #short paw, barks

test_x = [mystery1,mystery2,mystery3]
test_y = [0,1,1]    # this is the final answear, we have a dog,pig,pig


pred = model.predict(test_x)
print(pred)   ### will predict what animal is


correct = (pred == test_y).sum()
print(correct)  


taxa = accuracy_score(test_y,pred)
print("The accuracy is: %.2f" %(taxa*100) +"%") ### print the accurary of what it thought what animal was and what really was
