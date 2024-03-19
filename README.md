# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.

## Program:
```
## Developed by: SWETHA D
## RegisterNumber: 212223040222

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:

![267753589-cba641d7-4b64-474a-9df3-f8047b4ddc21](https://github.com/swetha23013979/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/153823422/2fee3680-b54e-4051-9bff-10be23be21f0)

![267753686-b69592e3-fb46-446d-87a4-60e8dabf45a1](https://github.com/swetha23013979/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/153823422/2f90b30d-7f8a-454a-83d0-e84ab8e4d071)

![267753782-196a08f0-0571-40f2-bfdf-b6e1d2b4fa8f](https://github.com/swetha23013979/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/153823422/e9fcb6ea-ee5c-4794-9d7d-016f26cb69be)

![267753891-3efb2a8c-6c60-4466-99b2-2c3c7b7a39b4](https://github.com/swetha23013979/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/153823422/633de827-c637-49e7-8465-78098c555bc6)

![267753963-37d05f23-2187-49d2-a871-7dbf5d7baca9](https://github.com/swetha23013979/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/153823422/059a0f67-4a5c-45ed-8555-a4d560a50188)

![267754049-d0b24ebb-4d7a-4956-b6e5-b87f65ccbeeb](https://github.com/swetha23013979/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/153823422/f60d2ed4-dc4e-4e66-900f-13e423aed591)

![267754328-81a5cd80-1fa0-48d8-a838-567b6e7a6676](https://github.com/swetha23013979/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/153823422/1eb39d00-14c8-4bb7-8061-5abde4960f69)

![267754448-1ca21819-8baa-4312-aae8-1b094fe75ea6](https://github.com/swetha23013979/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/153823422/5ab476d5-ad14-4fe1-96ea-a70772547cd7)

![267754513-675efabe-006d-463a-b5f0-0cc4354ca37a](https://github.com/swetha23013979/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/153823422/a8c6563e-3482-4b97-b8f1-370436d42b1d)

![267754597-be3ab929-d71c-492a-8adc-9a054cf08983](https://github.com/swetha23013979/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/153823422/d69fd152-8990-473c-8675-ad8d69294ae8)

![267754663-295b82c5-385c-4832-9d92-282a651946cb](https://github.com/swetha23013979/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/153823422/d493ead5-7d58-490e-b1f0-d8322c8f4529)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
