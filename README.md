# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values.


## Program:


Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: CHITTOOR SARAVANA MRUDHULA 


RegisterNumber:  212224040056



```
import pandas as pd
df = pd.read_csv("/content/Employee.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df['left'].value_counts())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['salary'] = le.fit_transform(df['salary'])

x = df[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]

print(x)
y = df['left']
print(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 30)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
print("Y Predicted : \n\n",y_pred)

from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy * 100:.2f}%")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```


## Output:

df.head()
![image](https://github.com/user-attachments/assets/97784ff2-4a8c-450e-a628-af499add1a61)

df.info()
![image](https://github.com/user-attachments/assets/3012d83a-8824-47d3-8358-a9df2970bd68)

df.isnull().sum()
![image](https://github.com/user-attachments/assets/c6a60fe4-9f9f-4eec-8662-2b539dd8a4fd)

df['left'].value_counts()
![image](https://github.com/user-attachments/assets/4c391b83-2d62-489c-971b-93840112c924)

```
print(x)
y = df['left']
print(y)
print("Y Predicted : \n\n",y_pred)
print(f"\nAccuracy : {accuracy * 100:.2f}%")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
![image](https://github.com/user-attachments/assets/d166875d-0fa5-4e84-b2e6-a85768bf8d43)

![image](https://github.com/user-attachments/assets/c6055a14-9c02-4e24-a062-edb08df5dab3)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
