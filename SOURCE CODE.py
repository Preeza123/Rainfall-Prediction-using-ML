#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import metrics 


# In[2]:


data = pd.read_csv('Rainfall_Data_LL.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


# data visualization
data[["SUBDIVISION","ANNUAL"]].groupby("SUBDIVISION").sum().sort_values(by=
'ANNUAL',ascending=False).plot(kind='barh',stacked=True, 
figsize=(15,10))
# displaying each state annual rainfall in mm
plt.xlabel("Rainfall in MM",size=12) 
plt.ylabel("Sub-Division",size=12) 
plt.title("Annual Rainfall v/s SubDivisions") 
plt.grid(axis="x",linestyle="-.") 
plt.show()


# In[6]:


# Rainfall over years
plt.figure(figsize=(15,8))
data.groupby("YEAR").sum()['ANNUAL'].plot(kind="line",color="r",marker=".") 
plt.xlabel("YEARS",size=12) 
plt.ylabel("RAINFALL IN MM",size=12) 
plt.grid(axis="both",linestyle="-.") 
plt.title("Rainfall over Years") 
plt.show()


# In[7]:


# Rainfall for monthly over years
data[['YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL','AUG', 'SEP', 
 'OCT', 'NOV', 
'DEC']].groupby("YEAR").sum().plot(kind="line",figsize=(18,8))
plt.xlabel("Year",size=13) 
plt.ylabel("Rainfall in MM",size=13) 
plt.title("Year v/s Rainfall in each month",size=20) 
plt.show()


# In[8]:


# grouping months for better analysis
data[['YEAR','Jan-Feb', 'Mar-May',  'June-September','Oct-Dec']].groupby("YEAR").sum().plot(figsize=(10,7))
plt.xlabel("Year",size=13) 
plt.ylabel("Rainfall in MM",size=13) 
plt.show()


# In[9]:


data[['SUBDIVISION', 'Jan-Feb', 'Mar-May', 'June-September','Oct-Dec']].groupby("SUBDIVISION").sum().plot(kind="barh",stacked=True,figsize=(16,8))
plt.xlabel("Rainfall in MM",size=12) 
plt.ylabel("Sub-Division",size=12) 
plt.grid(axis="x",linestyle="-.") 
plt.show()


# In[10]:


# ANALYSING States(TamilNadu and Rajasthan) by creating subgroups
TN = data.loc[((data['SUBDIVISION'] == 'Tamil Nadu'))]
TN.head()


# In[11]:


TN[['JAN', 'FEB', 'MAR', 'APR','MAY', 'JUN','JUL','AUG', 'SEP','OCT','NOV','DEC']].mean().plot(kind="bar",width=0.5,linewidth=2) 
plt.title("Tamil Nadu Rainfall v/s Months",size=20) 
plt.xlabel("Months",size=14) 
plt.ylabel("Rainfall in MM",size=14) 
plt.grid(axis="both",linestyle="-.") 
plt.show()


# In[12]:


TN.groupby("YEAR").sum()['ANNUAL'].plot(ylim=(50,1500),color='r',marker='o',linestyle='-',linewidth=2,figsize=(12,8));
plt.xlabel('Year',size=14) 
plt.ylabel('Rainfall in MM',size=14) 
plt.title('Tamil Nadu Annual Rainfall from Year 1901 to 2015',size=20) 
plt.grid()
plt.show()


# In[13]:


Rajasthan = data.loc[((data['SUBDIVISION'] == 'West Rajasthan') | (data['SUBDIVISION'] == 'East Rajasthan'))] 
Rajasthan.head()


# In[14]:


plt.figure(figsize=(10,6))
Rajasthan[['JAN', 'FEB', 'MAR', 'APR','MAY', 'JUN','JUL','AUG', 'SEP', 'OCT','NOV','DEC']].mean().plot(kind="bar",width=0.5,linewidth=2) 
plt.title("Rajasthan Rainfall v/s Months",size=20) 
plt.xlabel("Months",size=14) 
plt.ylabel("Rainfall in MM",size=14) 
plt.grid(axis="both",linestyle="-.") 
plt.show()
plt.figure(figsize=(10,6))
Rajasthan[['JAN', 'FEB', 'MAR', 'APR','MAY', 'JUN','JUL','AUG', 'SEP', 'OCT','NOV','DEC']].mean().plot(kind="bar",width=0.5,linewidth=2) 
plt.title("Rajasthan Rainfall v/s Months",size=20) 
plt.xlabel("Months",size=14) 
plt.ylabel("Rainfall in MM",size=14) 
plt.grid(axis="both",linestyle="-.") 
plt.show()


# In[15]:


Rajasthan = data.loc[((data['SUBDIVISION'] == 'West Rajasthan') | (data['SUBDIVISION'] == 'EAST RAJASTHAN'))] 
Rajasthan.head()


# In[16]:


plt.figure(figsize=(15,6))
sns.heatmap(data[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT',
'NOV','DEC','ANNUAL']].corr(),annot=True) 
plt.show()


# In[17]:


data["SUBDIVISION"].nunique() #output:- 36
group = data.groupby('SUBDIVISION')[['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']] 
data=group.get_group(('Tamil Nadu'))
data.head()


# In[21]:


df=data.melt(['YEAR']).reset_index() 
df.head()
df= df[['YEAR','variable','value']].reset_index().sort_values(by=['YEAR','index'])
df.head()
df.YEAR.unique()
df.columns=['Index','Year','Month','Avg_Rainfall'] 
df.head()


# In[22]:


Month_map={'JAN':1,'FEB':2,'MAR'
:3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9, 
 'OCT':10,'NOV':11,'DEC':12} 
df['Month']=df['Month'].map(Month_map)
df.head()


# In[23]:


df.drop(columns="Index",inplace=True)


# In[24]:


df.head(2)


# In[25]:


df.groupby("Year").sum().plot() 
plt.show()


# In[26]:


X=np.asanyarray(df[['Year','Month']]).astype('int') 
y=np.asanyarray(df['Avg_Rainfall']).astype('int') 
print(X.shape) 
print(y.shape) 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# In[27]:


from sklearn.linear_model import LinearRegression 
LR = LinearRegression() 
LR.fit(X_train,y_train) 


# In[28]:


y_train_predict=LR.predict(X_train) 
y_test_predict=LR.predict(X_test) 
print("-------Test Data--------") 
print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict)) 
print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict))) 
print("\n-------Train Data--------") 
print('MAE:', metrics.mean_absolute_error(y_train,y_train_predict)) 
print('MSE:', metrics.mean_squared_error(y_train, y_train_predict)) 
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict))) 
print("\n-----Training Accuracy-------") 
print(round(LR.score(X_train,y_train),3)*100) 
print("-----Testing Accuracy--------") 
print(round(LR.score(X_test,y_test),3)*100)


# In[29]:


from sklearn.linear_model import Ridge 
from sklearn.model_selection import GridSearchCV 
 
ridge=Ridge() 
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5) 
ridge_regressor.fit(X_train,y_train) 
print(ridge_regressor.best_params_) 
print(ridge_regressor.best_score_) 
print("Best Parameter for Ridge:",ridge_regressor.best_estimator_) 
ridge=Ridge(alpha=100.0)


# In[30]:


ridge.fit(X_train,y_train)


# In[31]:


y_train_predict=ridge.predict(X_train) 
y_test_predict=ridge.predict(X_test)


# In[32]:


from sklearn import metrics 
print("-------Test Data--------") 
print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict)) 
print('MSE:', metrics.mean_squared_error(y_test, y_test_predict)) 
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict))) 
print("\n-------Train Data--------") 
print('MAE:', metrics.mean_absolute_error(y_train,y_train_predict)) 
print('MSE:', metrics.mean_squared_error(y_train, y_train_predict)) 
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict))) 
print("\n-----Training Accuracy-------") 
print(round(ridge.score(X_train,y_train),3)*100) 
print("-----Testing Accuracy--------") 
print(round(ridge.score(X_test,y_test),3)*100)


# In[33]:


from sklearn.neighbors import KNeighborsRegressor 
from sklearn import metrics 
# create KNN regressor object
knn_regr = KNeighborsRegressor(n_neighbors=5) 
# fit the model with the training data
knn_regr.fit(X_train, y_train)
# predict on the test data
y_test_predict = knn_regr.predict(X_test)
y_train_predict = knn_regr.predict(X_train)
# print the evaluation metrics
print("-------Test Data--------") 
print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict)) 
print('MSE:', metrics.mean_squared_error(y_test, y_test_predict)) 
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))
print("\n-------Train Data--------") 
print('MAE:', metrics.mean_absolute_error(y_train,y_train_predict)) 
print('MSE:', metrics.mean_squared_error(y_train, y_train_predict)) 
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))
print("\n-----Training Accuracy-------") 
print(round(knn_regr.score(X_train,y_train),3)*100) 
print("-----Testing Accuracy--------") 
print(round(knn_regr.score(X_test,y_test),3)*100)


# In[34]:


# Random Forest Model
from sklearn.ensemble import RandomForestRegressor 
random_forest_model = RandomForestRegressor(max_depth=100, 
max_features='sqrt', min_samples_leaf=4, min_samples_split=10, 
n_estimators=800) 
random_forest_model.fit(X_train, y_train)
y_train_predict=random_forest_model.predict(X_train)
y_test_predict=random_forest_model.predict(X_test)
# Random Forest Model
print("-------Test Data--------") 
print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict)) 
print('MSE:', metrics.mean_squared_error(y_test, y_test_predict)) 
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))
print("\n-------Train Data--------") 
print('MAE:', metrics.mean_absolute_error(y_train,y_train_predict)) 
print('MSE:', metrics.mean_squared_error(y_train, y_train_predict)) 
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))
print("-----------Training Accuracy------------") 
print(round(random_forest_model.score(X_train,y_train),3)*100) 
print("-----------Testing Accuracy------------") 
print(round(random_forest_model.score(X_test,y_test),3)*100)



# In[ ]:





# In[ ]:




