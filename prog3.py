import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

data=pd.read_csv("prog3.csv")
print("the first 5 values of data is:\n", data.head())
X=data.iloc[:,:-1]
print("\n The first 5 values of train data is \n", X.head())
y=data.iloc[:,-1]
print("The first 5 values of train output is\n",y.head())
le_outlook=LabelEncoder()``
X.outlook=le_outlook.fit_transform(X.outlook)
le_temperature=LabelEncoder()
X.temperature=le_temperature.fit_transform(X.temperature)
le_humidity=LabelEncoder()
X.humidity=le_humidity.fit_transform(X.humidity)
le_windy=LabelEncoder()
X.windy=le_windy.fit_transform(X.windy)
print("\n Now the train dataset is:\n ",X.head())
le_Playtennis=LabelEncoder()
y=le_Playtennis.fit_transform(y)
print("\n Now the y data is \n ",y)

from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.40)
classifier=GaussianNB()
classifier.fit(X_train,y_train)
print("Now the test output is \n",y_test)
print("THe predicted output is ", classifier.predict(X_test))

from sklearn.metrics import accuracy_score
print("Accuracy is", accuracy_score(classifier.predict(X_test),y_test))