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

![image](https://user-images.githubusercontent.com/75626387/197673513-4dc2abfd-c5f2-47d7-82c5-b4012e62b2f8.png)


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
![image](https://user-images.githubusercontent.com/75626387/197673670-ea4f5bb7-25cc-4f2b-969b-2c6ae4e1267a.png)

* #### With Split

**Taking 70% Training and 30% Testing data**

```bash
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state=0)
```

![image](https://user-images.githubusercontent.com/75626387/196880664-5baafa5f-6f57-4245-969d-94c854502a5e.png)

**Taking 80% Training and 20% Testing data**

```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=0)
```
![image](https://user-images.githubusercontent.com/75626387/196880745-bf23b116-2eae-4043-8212-ec1648023802.png)

**Taking 50% Training and 50% Testing data**

```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=0)
```
![image](https://user-images.githubusercontent.com/75626387/196880801-f657e3aa-3353-4813-a488-c24215bca675.png)

### Multiple Regression

```bash
X=df.iloc[:,[1,2]].values
y=df.iloc[:,-1].values

fig = px.scatter_3d(df,x='population',y='gender',z='suicides_no')
fig.show()
```

![image](https://user-images.githubusercontent.com/75626387/196893751-c01c8752-06d9-4868-bea4-c17f65f95942.png)


* **Without Split**

R2 score: 0.224429

Mean square error: 63791.035

Root mean square error: 252.569

Mean absolute error: 13.070


* **With Split**

![image](https://user-images.githubusercontent.com/75626387/196894115-7702e1d5-e104-422d-99d1-cbccf8f37ba4.png)

On splitting the model into train and test data, we can observe that the R2 score has reduced. Out of all the splits 70:30 gives a better accuracy and thus has a better R2 score.

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
