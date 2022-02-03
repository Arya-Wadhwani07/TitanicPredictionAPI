import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
import pickle

def convert2(cols):
  age=cols[0]
  pclass=cols[1]
  if pd.isnull(age):
    if pclass==1:
      return 39
    elif pclass==2:
      return 30
    else:
      return 27
  else:
    return age

titanic_train = pd.read_csv(r'D:/Work Stuff/Python Programs/Titanic_Prediction/train.csv')
titanic_train["Age"]=titanic_train[["Age","Pclass"]].apply(convert2,axis=1)
titanic_train.drop('Cabin',axis=1,inplace=True)
titanic_train.dropna(inplace=True)

data=titanic_train.drop(['Name','Ticket'],axis=1)
data=pd.get_dummies(data=data,columns=['Sex','Embarked'])

X=data.drop('Survived',axis=1)
y=data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model=LogisticRegression()
model.fit(X_train,y_train)
pred=model.predict(X_test)

print(confusion_matrix(y_test,pred))

filename = 'Model.pckl' 
pickle.dump(model, open(filename, 'wb'))
