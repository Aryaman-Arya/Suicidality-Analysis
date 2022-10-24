#!/usr/bin/env python
# coding: utf-8

# # Suicidality Analysis for assorted demographic Mental Health Records
# 
# ### Aryaman-Arya
# ### Sanjam-Bedi

# In[123]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_regression
import statsmodels.api as sm


# In[12]:


data = pd.read_csv(r"D:\master.csv")


# In[13]:


data


# In[14]:


df = data.drop(columns=['country','country-year','age'])
print('\033[1m' + 'Columns in updated Dataframe :' + '\033[0m', len(df.columns))


# In[15]:


df


# In[16]:


from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()

df['generation'] = lc.fit_transform(df['generation'])
df['gender']=lc.fit_transform(df['gender'])


# In[17]:


df


# ### Exploratory Data Analysis

# In[18]:


df.isnull().sum()


# In[19]:


df.shape


# In[20]:


df.columns


# In[21]:


df.describe()


# In[22]:


df.dtypes


# In[23]:


sns.pairplot(df)


# In[24]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
plt.show()


# ### 2. Linear Regression
For performing linear regression, we will consider X as the 'population' and y as 'suicide_no'
# In[25]:


X=df.iloc[:,:1].values
y=df.iloc[:,-1].values


# ### 2.1 Without Split

# In[27]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)


# In[28]:


regressor.intercept_


# In[29]:


regressor.coef_


# In[30]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
print("Coeff : %.6f" % regressor.coef_)
print("Intercept : %.3f" % regressor.intercept_)
print("R2 score : %.6f" % r2_score(y, regressor.predict(X)))
print("MSE: %.3f" % mean_squared_error(y,regressor.predict(X)))
print("RMSE: %.3f" % sqrt(mean_squared_error(y, regressor.predict(X))))
print("MAE: %.3f" % sqrt(mean_absolute_error(y, regressor.predict(X))))


# In[31]:


fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x = df['population'], y = df['suicides_no'])
plt.xlabel("Population")
plt.ylabel("No of suicides")

plt.show()


# In[32]:


sns.lmplot(x='population',y='suicides_no',data=df,line_kws={'color': 'red'})
plt.xlabel('Population:  Independent variable')
plt.ylabel('No of suicides: Target variable')
plt.title('Population vs no of suicides');


# ### SPLITTING DATA INTO TRAINING AND TESTING DATA

# ####  A. TAKING 70% TRAINING AND 30%TESTING DATA

# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state=0)


# In[34]:


regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[35]:


regressor.intercept_


# In[36]:


regressor.coef_


# In[37]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
print("Coeff : %.6f" % regressor.coef_)
print("Intercept : %.3f" % regressor.intercept_)
print("R2 score : %.6f" % r2_score(y, regressor.predict(X)))
print("MSE: %.3f" % mean_squared_error(y,regressor.predict(X)))
print("RMSE: %.3f" % sqrt(mean_squared_error(y, regressor.predict(X))))
print("MAE: %.3f" % sqrt(mean_absolute_error(y, regressor.predict(X))))


# In[38]:


plt.figure(figsize=(8,5))
plt.scatter(X_train, y_train)
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('No of suicides vs Population')
plt.xlabel('Population')
plt.ylabel('No of suicides')
plt.show()


# #### B. TAKING 80% TRAINING DATA AND 20% TESTING DATA

# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=0)


# In[40]:


regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[41]:


regressor.intercept_


# In[42]:


regressor.coef_


# In[43]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
print("Coeff : %.6f" % regressor.coef_)
print("Intercept : %.3f" % regressor.intercept_)
print("R2 score : %.6f" % r2_score(y, regressor.predict(X)))
print("MSE: %.3f" % mean_squared_error(y,regressor.predict(X)))
print("RMSE: %.3f" % sqrt(mean_squared_error(y, regressor.predict(X))))
print("MAE: %.3f" % sqrt(mean_absolute_error(y, regressor.predict(X))))


# In[44]:


plt.figure(figsize=(8,5))
plt.scatter(X_train, y_train)
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('No of suicides vs Population')
plt.xlabel('Population')
plt.ylabel('No of suicides')
plt.show()


# #### C. TAKING 50% TRAINING AND 50% TESTING DATA

# In[45]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=0)


# In[46]:


regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[47]:


lr = LinearRegression()


# In[48]:


regressor.intercept_


# In[49]:


regressor.coef_


# In[50]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
print("Coeff : %.6f" % regressor.coef_)
print("Intercept : %.3f" % regressor.intercept_)
print("R2 score : %.6f" % r2_score(y, regressor.predict(X)))
print("MSE: %.3f" % mean_squared_error(y,regressor.predict(X)))
print("RMSE: %.3f" % sqrt(mean_squared_error(y, regressor.predict(X))))
print("MAE: %.3f" % sqrt(mean_absolute_error(y, regressor.predict(X))))


# In[51]:


plt.figure(figsize=(8,5))
plt.scatter(X_train, y_train)
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('No of suicides vs Population')
plt.xlabel('Population')
plt.ylabel('No of suicides')
plt.show()


# In[52]:


import numpy as np
def predicted_y(weight,x,intercept):
    y_lst=[]
    for i in range(len(x)):
        y_lst.append(weight@x[i]+intercept)
    return np.array(y_lst)
    

# linear loss
def loss(y,y_predicted):
    n=len(y)
    s=0
    for i in range(n):
        s+=(y[i]-y_predicted[i])**2
    return (1/n)*s

#derivative of loss w.r.t weight
def dldw(x,y,y_predicted):
    s=0
    n=len(y)
    for i in range(n):
        s+=-x[i]*(y[i]-y_predicted[i])
    return (2/n)*s
    

# derivative of loss w.r.t bias
def dldb(y,y_predicted):
    n=len(y)
    s=0
    for i in range(len(y)):
        s+=-(y[i]-y_predicted[i])
    return (2/n) * s

# gradient function
def gradient_descent(x,y,epoch,learning_rate,color):
    weight_vector=np.random.randn(x.shape[1])
    intercept=0
    #epoch = 2000
    n = len(x)
    linear_loss=[]
    #learning_rate = 0.00002

    for i in range(epoch):
        
        
        y_predicted = predicted_y(weight_vector,x,intercept)
        
        weight_vector = weight_vector - learning_rate *dldw(x,y,y_predicted) 
        
        
        intercept = intercept - learning_rate * dldb(y,y_predicted)
        linear_loss.append(loss(y,y_predicted))
    
    plt.plot(np.arange(1,epoch),linear_loss[1:],color=str(color))
    plt.xlabel("Number of epoch")
    plt.ylabel("Loss")
    
    return weight_vector,intercept


# In[53]:


x_sr = df.iloc[: , :1].values
y = df.iloc[: , -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_sr, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_transform=sc.fit_transform(x_train)


# In[54]:


##Gradient Descent Method ( L = 0.1. 001, 0.5, 0.05, 1)
w,b=gradient_descent(X_transform,y_train,10,0.1,'blue')
w,b=gradient_descent(X_transform,y_train,10,0.001,'green')
w,b=gradient_descent(X_transform,y_train,10,0.5,'black')
w,b=gradient_descent(X_transform,y_train,10,1,'red')
w,b=gradient_descent(X_transform,y_train,10,0.05,'pink')


# In[55]:


for i in [0.001, 0.05, 0.1, 0.5, 0.75, 0.9, 0.95, 0.99, 1, 1.1, 1.5, 2, 2.5]:
#for i in [0.9]:
  w,b=gradient_descent(X_transform,y_train,30,i,'blue')
  def predict(inp):
      y_lst=[]
      for i in range(len(inp)):
          y_lst.append(w@inp[i]+b)
      return np.array(y_lst)

  X_test=sc.fit_transform(x_test)
  y_pred=predict(X_test)

  from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
  from math import sqrt
  print(f"learning rate {i}")
  print("R2 score : %.3f" % r2_score(y_test, y_pred))
  print("MSE: %.3f" % mean_squared_error(y_test, y_pred))
  print("RMSE: %.3f" % sqrt(mean_squared_error(y_test, y_pred)))
  print("MAE: %.3f \n" % sqrt(mean_absolute_error(y_test, y_pred)))


# In[56]:


for i in [0.0005,0.001, 0.05, 0.1, 0.5, 0.75, 0.9, 0.95, 0.99, 1, 1.1, 1.5, 2]:
#for i in [0.9]:
  w,b=gradient_descent(X_transform,y_train,30,i,'blue')
  def predict(inp):
      y_lst=[]
      for i in range(len(inp)):
          y_lst.append(w@inp[i]+b)
      return np.array(y_lst)

  X_test=sc.fit_transform(x_test)
  y_pred=predict(X_test)

  from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
  from math import sqrt
  print(f"learning rate {i}")
  print("R2 score : %.3f" % r2_score(y_test, y_pred))
  print("MSE: %.3f" % mean_squared_error(y_test, y_pred))
  print("RMSE: %.3f" % sqrt(mean_squared_error(y_test, y_pred)))
  print("MAE: %.3f \n" % sqrt(mean_absolute_error(y_test, y_pred)))


# In[57]:


for i in [0.001, 0.05, 0.1, 0.5, 0.75, 0.9, 0.95, 0.99, 1, 1.1, 1.5, 2, 2.5]:
#for i in [0.9]:
  w,b=gradient_descent(X_transform,y_train,30,i,'blue')
  def predict(inp):
      y_lst=[]
      for i in range(len(inp)):
          y_lst.append(w@inp[i]+b)
      return np.array(y_lst)

  X_test=sc.fit_transform(x_test)
  y_pred=predict(X_test)

  from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
  from math import sqrt
  print(f"learning rate {i}")
  print("R2 score : %.3f" % r2_score(y_test, y_pred))
  print("MSE: %.3f" % mean_squared_error(y_test, y_pred))
  print("RMSE: %.3f" % sqrt(mean_squared_error(y_test, y_pred)))
  print("MAE: %.3f \n" % sqrt(mean_absolute_error(y_test, y_pred)))


# ## Multiple Regression

# In[58]:


X=df.iloc[:,[1,2]].values
y=df.iloc[:,-1].values


# In[59]:


fig = px.scatter_3d(df,x='population',y='gender',z='suicides_no')
fig.show()


# ### Without Splitting

# In[61]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)


# In[62]:


regressor.intercept_


# In[63]:


regressor.coef_


# In[64]:


regressor.predict(X)


# In[65]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt


print("R2 score : %.6f" % r2_score(y, regressor.predict(X)))
print("MSE: %.3f" % mean_squared_error(y,regressor.predict(X)))
print("RMSE: %.3f" % sqrt(mean_squared_error(y, regressor.predict(X))))
print("MAE: %.3f" % sqrt(mean_absolute_error(y, regressor.predict(X))))


# ### Splitting Training and Testing Data

# ### Taking 70 as training and 30 as testing

# In[66]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state=0)


# In[67]:


regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[68]:


regressor.intercept_


# In[69]:


regressor.coef_


# In[70]:


print("R2 score : %.3f" % r2_score(y_test, regressor.predict(X_test)))
print("MSE: %.3f" % mean_squared_error(y_test, regressor.predict(X_test)))
print("RMSE: %.3f" % sqrt(mean_squared_error(y_test, regressor.predict(X_test))))
print("MAE: %.3f" % sqrt(mean_absolute_error(y_test, regressor.predict(X_test))))


# ### Taking 80% training and 20% testing

# In[71]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=0)


# In[72]:


regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[73]:


regressor.intercept_


# In[74]:


regressor.coef_


# In[75]:


print("R2 score : %.3f" % r2_score(y_test, regressor.predict(X_test)))
print("MSE: %.3f" % mean_squared_error(y_test, regressor.predict(X_test)))
print("RMSE: %.3f" % sqrt(mean_squared_error(y_test, regressor.predict(X_test))))
print("MAE: %.3f" % sqrt(mean_absolute_error(y_test, regressor.predict(X_test))))


# ### Taking 50 %training and 50% testing data

# In[76]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=0)


# In[77]:


regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[78]:


regressor.intercept_


# In[79]:


regressor.coef_


# In[80]:


print("R2 score : %.3f" % r2_score(y_test, regressor.predict(X_test)))
print("MSE: %.3f" % mean_squared_error(y_test, regressor.predict(X_test)))
print("RMSE: %.3f" % sqrt(mean_squared_error(y_test, regressor.predict(X_test))))
print("MAE: %.3f" % sqrt(mean_absolute_error(y_test, regressor.predict(X_test))))


# ## Polynomial regression

# ### Without splitting

# In[81]:


X=df.iloc[:,:1].values
y=df.iloc[:,-1].values


# In[82]:


from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X,y)


# In[83]:


from sklearn.preprocessing import PolynomialFeatures
polynom=PolynomialFeatures(degree=1)
X_polynom=polynom.fit_transform(X_train)
X_polynom


# In[84]:


PolyRegr=LinearRegression()
PolyRegr.fit(X_polynom,y_train)


# In[85]:


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 1)
X_poly = poly.fit_transform(X)
X_poly


# In[86]:


polyreg = LinearRegression()
polyreg.fit(X_poly, y)


# In[87]:


#polynomial intercept and coefficient
polyreg.intercept_


# In[88]:


polyreg.coef_


# In[89]:


print("Intercept: %0.3f" % PolyRegr.intercept_)
print("R2 score : %.3f" % r2_score(y_test, PolyRegr.predict(polynom.fit_transform(X_test))))
print("MSE: %.3f" % mean_squared_error(y_test, PolyRegr.predict(polynom.fit_transform(X_test))))
print("RMSE: %.3f" % sqrt(mean_squared_error(y_test, PolyRegr.predict(polynom.fit_transform(X_test)))))
print("MAE: %.3f" % sqrt(mean_absolute_error(y_test, PolyRegr.predict(polynom.fit_transform(X_test)))))


# In[90]:


plt.scatter(X,y,color='red')
plt.plot(X,polyreg.predict(poly.fit_transform(X)),color='blue')
plt.title("Polynomial regression")
plt.xlabel("Population")
plt.ylabel("No of suicides")


# ### 2. SPLITTING TRAINING AND TESTING DATA

# In[91]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state=0)


# In[92]:


linreg=LinearRegression()
linreg.fit(X_train,y_train)


# In[93]:


poly = PolynomialFeatures(degree = 1)
X_poly = poly.fit_transform(X_train)
X_poly


# In[94]:


polyreg = LinearRegression()
polyreg.fit(X_poly, y_train)


# In[95]:


polyreg.intercept_


# In[96]:


polyreg.coef_


# In[97]:


plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,polyreg.predict(poly.fit_transform(X_train)),color='blue')
plt.title("Polynomial regression")
plt.xlabel("Population")
plt.ylabel("suicide_no")


# In[98]:


print("Intercept: %0.3f" % PolyRegr.intercept_)


# In[99]:


X = df[['population']]
y= df['suicides_no']


# In[100]:


from sklearn.linear_model import Ridge
ridgeReg = Ridge(alpha = 0.05, normalize = True)
ridgeReg.fit(X,y)
pred = ridgeReg.predict(X)
mse = np.mean((pred-y)**2)
mse


# In[101]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
print(ridgeReg.coef_)
print("Intercept :%0.3f"% ridgeReg.intercept_)
print("R2 score: %0.3f"% r2_score(y, ridgeReg.predict(X)))
print("MSE: %0.3f"% mean_squared_error(y, ridgeReg.predict(X)))
print("RMSE: %0.3f"% math.sqrt(mean_squared_error(y, ridgeReg.predict(X))))
print("MAE: %0.3f"% math.sqrt(mean_absolute_error(y, ridgeReg.predict(X))))


# In[102]:


import statsmodels.api as sm
X_train_sm=sm.add_constant(X_train)
lr=sm.OLS(y_train,X_train_sm).fit()
lr.params


# In[103]:


print(lr.ssr)


# In[104]:


m=len(y)
p=1
hat_sigma_squared=(1/(m-p-1))*76933400.19333948
Cp=(1/m)*(76933400.19333948+2*1* hat_sigma_squared)
Cp


# In[105]:


lr.summary()


# In[106]:


import numpy as np
def predicted_y(weight,x,intercept):
    y_lst=[]
    for i in range(len(x)):
        y_lst.append(weight@x[i]+intercept)
    return np.array(y_lst)
    

# linear loss
def loss(y,y_predicted):
    n=len(y)
    s=0
    for i in range(n):
        s+=(y[i]-y_predicted[i])**2
    return (1/n)*s

#derivative of loss w.r.t weight
def dldw(x,y,y_predicted):
    s=0
    n=len(y)
    for i in range(n):
        s+=-x[i]*(y[i]-y_predicted[i])
    return (2/n)*s
    

# derivative of loss w.r.t bias
def dldb(y,y_predicted):
    n=len(y)
    s=0
    for i in range(len(y)):
        s+=-(y[i]-y_predicted[i])
    return (2/n) * s

# gradient function
def gradient_descent(x,y,epoch,learning_rate,color):
    weight_vector=np.random.randn(x.shape[1])
    intercept=0
    #epoch = 2000
    n = len(x)
    linear_loss=[]
    #learning_rate = 0.00002

    for i in range(epoch):
        
        
        y_predicted = predicted_y(weight_vector,x,intercept)
        
        weight_vector = weight_vector - learning_rate *dldw(x,y,y_predicted) 
        
        
        intercept = intercept - learning_rate * dldb(y,y_predicted)
        linear_loss.append(loss(y,y_predicted))
        
    plt.plot(np.arange(1,epoch),linear_loss[1:],color=str(color))
    plt.xlabel("number of epoch")
    plt.ylabel("loss")
    
    return weight_vector,intercept


# In[107]:


##Gradient Descent Method ( L = 0.1. 001, 0.5, 0.05, 1)
w,b=gradient_descent(X_train,y_train,500,1,'RED')
print("coeff:",w)
print("intercept:",b)


# In[108]:


w,b=gradient_descent(X_train,y_train,1000,1,'RED')


# In[109]:


w,b=gradient_descent(X_train,y_train,1000,0.05,'black')


# In[110]:


w,b=gradient_descent(X_train,y_train,1000,0.001,'blue')


# In[111]:


w,b=gradient_descent(X_train,y_train,1000,0.0001,'green')


# In[117]:


w,b=gradient_descent(X_train,y_train,1000,0.00001,'yellow')


# In[118]:


import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[119]:


import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
print(ridgeReg.coef_)
print("Intercept: %0.3f" % ridgeReg.intercept_)
print("R2 score : %.3f" % r2_score(y, ridgeReg.predict(X)))
print("MSE: %.3f" % mean_squared_error(y, ridgeReg.predict(X)))
print("RMSE: %.3f" % math.sqrt(mean_squared_error(y, ridgeReg.predict(X))))
print("MAE: %.3f" % math.sqrt(mean_absolute_error(y, ridgeReg.predict(X))))


# In[120]:


from sklearn.linear_model import Lasso
las = Lasso(alpha = 0.05, normalize = True)
las.fit(X, y)
pred = las.predict(X)
mse = np.mean((pred - y)**2)
mse


# In[121]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
print(las.coef_)
print("Intercept: %0.3f" % las.intercept_)
print("R2 score : %.3f" % r2_score(y, las.predict(X)))
print("MSE: %.3f" % mean_squared_error(y, las.predict(X)))
print("RMSE: %.3f" % math.sqrt(mean_squared_error(y, las.predict(X))))
print("MAE: %.3f" % math.sqrt(mean_absolute_error(y, las.predict(X))))

