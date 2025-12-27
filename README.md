# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("income.csv",na_values=[ " ?"])
data
```
<img width="1686" height="742" alt="image" src="https://github.com/user-attachments/assets/ad670197-7bc6-451f-beb9-d4890da5a1cf" />


```
data.isnull().sum()
```
<img width="206" height="600" alt="image" src="https://github.com/user-attachments/assets/86282b93-f2c3-4054-9838-c527d804138a" />


```
missing=data[data.isnull().any(axis=1)]
missing
```
<img width="1685" height="732" alt="image" src="https://github.com/user-attachments/assets/c5a4cd8a-d415-4aec-aaa6-72c31e770f72" />


```
data2=data.dropna(axis=0)
data2
```
<img width="1673" height="743" alt="image" src="https://github.com/user-attachments/assets/c9b53d97-ee0a-428a-934d-cf503a8f3eb8" />


```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/c26296a0-3f88-4bea-bf79-02b5ae669153)

```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
<img width="444" height="509" alt="image" src="https://github.com/user-attachments/assets/0c92a9ad-8193-47b8-aa77-be3b9949b878" />


```
data2
```
<img width="1696" height="571" alt="image" src="https://github.com/user-attachments/assets/829ee089-f0d9-41fa-aa22-901c74edc8b8" />


```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
<img width="1744" height="605" alt="image" src="https://github.com/user-attachments/assets/a9e09ebb-6405-4102-90a0-954b46b5612a" />

```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/f2a2209e-dd66-41b9-8610-3e75f516f6cb)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/d20c23e6-83a2-4190-9cf6-9b0f31be09d0)

```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/2e35eac6-5a50-443c-994e-f76f201a97fe)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/b3b040cd-c01e-41b1-b016-5a997c04d144)

```

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/cb077c23-9c1e-4d18-9c88-467ce8803cbb)

```
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/da211449-6ff6-4da2-af63-c0c42aa74a5a)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/a843759f-5e5c-41fa-af77-0da66b255bc4)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/be2b2b8a-a9c6-45fa-903b-5b81c1fae78f)
```
data.shape
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/33382151-f32e-4a5a-9b74-1de057499237)


```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/b193f7fe-d80b-45b7-a9f4-1bea3fcec899)


```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/0d4792c8-f1ea-44c6-bce6-70f725363a75)
```
tips.time.unique()
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/98f7d5ab-4712-4844-812c-4715c65b96a2)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/8ca322ba-d9c9-4b94-b50e-876db6259e4f)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/74662e12-9184-43e6-8754-32337b8aa010)
# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
save the data to a file is been executed.

