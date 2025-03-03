import pandas as pd
msg= pd.read_csv("prog4.csv",names=['message','label'])
print("total instances of dataset'",msg.shape[0])

msg['labelnum']=msg.label.map({'pos':1,'neg':0})

x=msg.message
y=msg.labelnum

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y)

from sklearn.feature_extraction.text import CountVectorizer

count_v=CountVectorizer()
xtrain_dm=count_v.fit_transform(xtrain)
xtest_dm=count_v.transform(xtest)

df=pd.DataFrame(xtrain_dm.toarray(),columns=count_v.get_feature_names_out())
print(df[0:5])

from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
clf.fit(xtrain_dm,ytrain)
pred=clf.predict(xtest_dm)

for doc,P in zip(xtrain,pred):
    P='pos' if P==1 else 'neg'
    print("%s->%s"%(doc,P))

from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score

print("Accuracy matrix",accuracy_score(ytest,pred))
print("recall",recall_score(ytest,pred))
print("precision",precision_score(ytest,pred))
print("Confusion matrix",confusion_matrix(ytest,pred))
