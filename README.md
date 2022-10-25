# Suicidality Analysis for assorted demographic Mental Health Records

Analysed the risk of suicides for varying compositions of the human population by implementing possible regression models to get the most accurate model on the basis of their Mental Health Records.

## Tech Stack

* **[Pandas](https://pandas.pydata.org/docs/getting_started/install.html)** 

* **[NumPy](https://numpy.org/install/)** 

* **[Matplotlib](https://matplotlib.org/stable/users/installing/index.html)** 

* **[Seaborn](https://seaborn.pydata.org/installing.html)** 

* **[Scikit-learn](https://scikit-learn.org/stable/install.html)** 
## Implementation

The dataset that we have selected for our project” Mental Health Survey” is the Analysis of suicidal cases from the year 1987 to 2000 of various countries. the dataset has various parameters including Country, generation, gender, population, GDP of the country and number of suicide cases. By collecting information about past suicidal data and analysing them, we can create a model that could be used to curb this problem by providing mental health training to the people in maximum need.

![image](https://user-images.githubusercontent.com/75626387/197673456-9efd1c8d-043b-4cc5-b445-e01bb2296274.png)

The parameters that have a large correlation value, are largely correlated with each other or the dependent variable is largely affected by the independent variable.

```bash
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
plt.show()
```

![image](https://user-images.githubusercontent.com/75626387/197766442-63467ff8-7e5e-4332-920f-304be9e3160f.png)


Through this heatmap, we observed that the parameter suicides_no has larger
correlation with population and gender as compared to the other parameters.
Thus, we considered these parameters in order to create our model.

### Linear Regression

* #### Without Split


```bash
sns.lmplot(x='population',y='suicides_no',data=df,line_kws={'color': 'red'})
plt.xlabel('Population:  Independent variable')
plt.ylabel('No of suicides: Target variable')
plt.title('Population vs no of suicides');
```
![image](https://user-images.githubusercontent.com/75626387/197766605-43b7c660-b78f-424c-884e-793658de7cdc.png)

* #### With Split

**Taking 70% Training and 30% Testing data**

```bash
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state=0)
```

![image](https://user-images.githubusercontent.com/75626387/197766678-2ce9493d-6561-4380-a313-8f7ff0b6eaa1.png)

**Taking 80% Training and 20% Testing data**

```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=0)
```
![image](https://user-images.githubusercontent.com/75626387/197766723-6f9b571f-9441-471d-ae57-59e90d632029.png)

**Taking 50% Training and 50% Testing data**

```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=0)
```
![image](https://user-images.githubusercontent.com/75626387/197766810-61570cbf-d23f-4619-8962-fde28b8e47bf.png)

### Multiple Regression

```bash
X=df.iloc[:,[1,2]].values
y=df.iloc[:,-1].values

fig = px.scatter_3d(df,x='population',y='gender',z='suicides_no')
fig.show()
```

![newplot](https://user-images.githubusercontent.com/75626387/197767329-8fdfa562-6ec2-4892-aaf2-dd6564108573.png)


* **Without Split**

R2 score: 0.025905

MSE(Mean square error): 792583.054

RMSE(Root mean square error): 890.271

MAE(Mean absolute error): 18.017


* **With Split**

**Taking 70% Training and 30% Testing data**

R2 score : -0.236
MSE: 476057.532
RMSE: 689.969
MAE: 19.888

**Taking 80% Training and 20% Testing data**

R2 score : -0.256
MSE: 625660.232
RMSE: 790.987
MAE: 20.844

**Taking 50% Training and 50% Testing data**

R2 score : -0.178
MSE: 302277.129
RMSE: 549.797
MAE: 17.488

On splitting the model into train and test data, we can observe that the R2 score has reduced. Out of all the splits 80:20 gives a better accuracy and thus has a better R2 score.

### Polynomial Regression

We have performed the polynomial regression for degree 1, 2 and 3.

* **Without split**

**Degree 1**

![image](https://user-images.githubusercontent.com/75626387/197014032-11f442b5-3930-4194-b12b-f59a27e397d8.png)


**Degree 2**

![image](https://user-images.githubusercontent.com/75626387/197014049-9c85b777-a358-442a-9e43-92d1cb5bb7c4.png)


**Degree 3**

![image](https://user-images.githubusercontent.com/75626387/197014080-08e84ffc-acb7-4993-9db0-9aa244a22c1f.png)

Here we can observe that Polynomial regression on degree 3 gives a larger R2 score than on degree 1 and 2 for this dataset.

![image](https://user-images.githubusercontent.com/75626387/197014104-91e04762-3fd2-499e-8a96-2a7d80f6f4df.png)

* **With split**

**Degree 1**

![image](https://user-images.githubusercontent.com/75626387/197014124-b3ed9537-7f39-4db4-88c9-a935a1dc1d48.png)

![image](https://user-images.githubusercontent.com/75626387/197014149-8eb96357-3e01-4001-b74b-e7f4d473d33f.png)

**Degree 2**

![image](https://user-images.githubusercontent.com/75626387/197014183-66c15d9b-bb71-414a-8b39-f1797877c102.png)

![image](https://user-images.githubusercontent.com/75626387/197014209-5680d0e2-4b52-4afb-baaa-40c6a4ffa871.png)

### Regularization

* **Ridge Regression**

Using ridge regression, we are able to curb the overfitting of the model and reduce the complexity. Comparing the ridge regression with predictors 1 and 2, here are the results that we get.

![image](https://user-images.githubusercontent.com/75626387/197011286-693f5d18-e23a-4baf-9763-c9fbb72cb991.png)

Here, we can observe that the R2 score with 2 predictors is greater than that of 1 and thus 2 predictors will give more accurate results than 1 predictor. The 2 predictors here are population and gender.

* **Lasso Regression**

![image](https://user-images.githubusercontent.com/75626387/197011494-16cbadd8-9e41-44bb-b1ee-7d3473ce421c.png)

Lasso and ridge regression both give almost similar results, the reason being both are used to curb the overfitting and make the model less complex.

After applying all the statistical learning techniques to build the machine learning model, we conclude that among all the regression techniques, polynomial regression with degree 1 gives the most accurate results on splitting the dataset into training and testing with the ratio of 80:20. It gives an accuracy of 33%. Our model gives a lesser accuracy because the dataset that we have chosen has very large-scale values of the parameter population and so it does not give very accurate results. On applying the ridge and lasso regression techniques and curbing the overfitting, the model finally gives an accuracy of 39%.

## Authors

- Aryaman : [@Aryaman-Arya](https://github.com/Aryaman-Arya)
- Sanjam Kaur Bedi : [@Sanjam-Bedi](https://github.com/Sanjam-Bedi)
