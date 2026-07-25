import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#data loading
data=pd.read_csv('heart.csv')
print(data.info())



#data assingning x,y
x=data.drop('target',axis=1)
y=data['target']


#data splitting
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


#model selection RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)

#model training
model.fit(x_train,y_train)

#model prediction
prediction=model.predict(x_test)
print("predicted values:",prediction)
print("actual values:",y_test.values)

#model evaluation
acc=accuracy_score(y_test,prediction)
print("Accuracy:",acc)

#model deploying
import joblib
joblib.dump(model,'heart_disease_model.pkl')
print("Model saved as heart_disease_model.pkl")
