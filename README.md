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
 from scipy import stats
 import numpy as np
 df=pd.read_csv("/content/bmi.csv")
 df.head()
```
![Screenshot 2025-04-19 191016](https://github.com/user-attachments/assets/34363739-bd78-4273-98a3-3626ad86acbd)
```
df.dropna()
```
![Screenshot 2025-04-19 190838](https://github.com/user-attachments/assets/368937a2-a9cc-4262-8c71-22afa2e46ba5)
```
max_val=np.max(np.abs(df[['Height','Weight']]))
 max_val
```
![Screenshot 2025-04-19 190628](https://github.com/user-attachments/assets/8a369e38-2bd2-4de4-b729-745928599117)
```
from sklearn.preprocessing import StandardScaler
 sc=StandardScaler()
 df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
 df.head(10)
```
![Screenshot 2025-04-19 190717](https://github.com/user-attachments/assets/7b1e5fdd-9f55-4de7-ba56-cd1f43701d6a)
```
from sklearn.preprocessing import Normalizer
 nm=Normalizer()
 df[['Height','Weight']]=nm.fit_transform(df[['Height','Weight']])
 df
```
![Screenshot 2025-04-19 190955](https://github.com/user-attachments/assets/8b9ed9e8-92ec-435e-9b4c-2b174eb0a252)

```
rom sklearn.preprocessing import MaxAbsScaler
 mas=MaxAbsScaler()
 df[['Height','Weight']]=mas.fit_transform(df[['Height','Weight']])
 df
```
 from sklearn.preprocessing import RobustScaler
 rs=RobustScaler()
 df[['Height','Weight']]=rs.fit_transform(df[['Height','Weight']])
 df.head(5)
 ```

![Screenshot 2025-04-19 191036](https://github.com/user-attachments/assets/2b243774-579e-4874-bfa9-ec727d3a304f)
```

![image](https://github.com/user-attachments/assets/afacc081-044f-45f6-a6d0-f4166bd0fdad)
```
 contingency_table=pd.crosstab(tips['sex'],tips['time'])
 contingency_table
```

![Screenshot 2025-04-19 191055](https://github.com/user-attachments/assets/94aa9fa8-b8b0-469a-96cc-3864036971fb)
```
 chi2,p,_,_=chi2_contingency(contingency_table)
 print('Chi-square statistic:',chi2)
 print('p-value:',p)
```

![Screenshot 2025-04-19 191117](https://github.com/user-attachments/assets/8033275a-d82f-43e6-9995-d0ee08fb98c8)
```
 from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_cla
 data={'Feature1' : [1,2,3,4,5],'Feature2' : ['A','B','C','A','B'],'Feature3
 df=pd.DataFrame(data)
 df
```

![Screenshot 2025-04-19 191141](https://github.com/user-attachments/assets/ea837d4f-7251-46da-8110-8c337cabe732)
```
x=df[['Feature1','Feature3']]
 y=df['Target']
 selector=SelectKBest(score_func=mutual_info_classif,k=1)
 x_new=selector.fit_transform(x,y)
 print('Selected features:',x_new)
```

![Screenshot 2025-04-19 191257](https://github.com/user-attachments/assets/31a2d6f7-4ada-4185-99fb-0af3dbac88d7)
```
 selectedFeatureIndices=selector.get_support(indices=True)
 selectedFeatures=x.columns[selectedFeatureIndices]
 print('Selected features:',selectedFeatures)
```


![Screenshot 2025-04-19 191319](https://github.com/user-attachments/assets/4e3b7d66-0b4c-40cc-b24e-a6718b5598fd)


# RESULT:
      Feature Scaling and Feature
 Selection process has been successfully
 performed on the data set.
