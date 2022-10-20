# Suicidality Analysis for assorted demographic Mental Health Records

Analysed the risk of suicides for varying compositions of the human population by implementing possible regression models to get the most accurate model on the basis of their Mental Health Records.

## Tech Stack

**[Pandas](https://pandas.pydata.org/docs/getting_started/install.html)** 

**[NumPy](https://numpy.org/install/)** 

**[Matplotlib](https://matplotlib.org/stable/users/installing/index.html)** 

**[Seaborn](https://seaborn.pydata.org/installing.html)** 

**[Scikit-learn](https://scikit-learn.org/stable/install.html)** 
## Implementation

To deploy this project run

```bash
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
plt.show()
```

![image](https://user-images.githubusercontent.com/75626387/196880241-a5202c4c-353b-436a-9630-f7f60d330d70.png)


Through this heatmap, we observed that the parameter suicides_no has larger
correlation with population and gender as compared to the other parameters.
Thus, we considered these parameters in order to create our model.

### Linear Regression

#### Without Split


```bash
sns.lmplot(x='population',y='suicides_no',data=df,line_kws={'color': 'red'})
plt.xlabel('Population:  Independent variable')
plt.ylabel('No of suicides: Target variable')
plt.title('Population vs no of suicides');
```
![image](https://user-images.githubusercontent.com/75626387/196880556-b87f429c-f4c8-497c-a3d8-a73589c9038f.png)

#### With Split

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

**Gradient Descent**

##Plot 3

### Multiple Regression

```bash
X=df.iloc[:,[1,2]].values
y=df.iloc[:,-1].values

fig = px.scatter_3d(df,x='population',y='gender',z='suicides_no')
fig.show()
```

![image](https://user-images.githubusercontent.com/75626387/196893751-c01c8752-06d9-4868-bea4-c17f65f95942.png)


**Without Split**

R2 score: 0.224429

Mean square error: 63791.035

Root mean square error: 252.569

Mean absolute error: 13.070


**With Split**

![image](https://user-images.githubusercontent.com/75626387/196894115-7702e1d5-e104-422d-99d1-cbccf8f37ba4.png)

On splitting the model into train and test data, we can observe that the R2 score has reduced. Out of all the splits 70:30 gives a better accuracy and thus has a better R2 score.

### Polynomial Regression

We have performed the polynomial regression for degree 1, 2 and 3.

**Without split**

**Degree 1**

![image](https://user-images.githubusercontent.com/75626387/197009979-6be5dde6-b4c2-40a1-9aa8-a0447bdf4631.png)


**Degree 2**

![image](https://user-images.githubusercontent.com/75626387/197010015-05a648a1-9d0f-408d-bd96-68f2debfa266.png)


**Degree 3**

![image](https://user-images.githubusercontent.com/75626387/197010037-cd9894ab-f77c-41bd-9525-04d6568ba69c.png)

Here we can observe that Polynomial regression on degree 3 gives a larger R2 score than on degree 1 and 2 for this dataset.

![image](https://user-images.githubusercontent.com/75626387/197010154-6f836630-c0c3-4e6b-b2bd-0f004c341417.png)

**With split**

**Degree 1**

![image](https://user-images.githubusercontent.com/75626387/197010573-dc168e31-7f59-4466-a7e8-40efbf95edf6.png)

![image](https://user-images.githubusercontent.com/75626387/197010627-bc86ce4f-4804-4a80-aa04-ac0e67b6d3f3.png)

**Degree 2**

![image](https://user-images.githubusercontent.com/75626387/197010742-26ea7935-4de1-483a-b6a9-441b3e761f20.png)

![image](https://user-images.githubusercontent.com/75626387/197010810-134c864d-7973-4ff8-a879-7a02d56bbc9b.png)

### Regularization

**Ridge Regression**

Using ridge regression, we are able to curb the overfitting of the model and reduce the complexity. Comparing the ridge regression with predictors 1 and 2, here are the results that we get.

![image](https://user-images.githubusercontent.com/75626387/197011286-693f5d18-e23a-4baf-9763-c9fbb72cb991.png)

Here, we can observe that the R2 score with 2 predictors is greater than that of 1 and thus 2 predictors will give more accurate results than 1 predictor. The 2 predictors here are population and gender.

**Lasso Regression**

![image](https://user-images.githubusercontent.com/75626387/197011494-16cbadd8-9e41-44bb-b1ee-7d3473ce421c.png)

Lasso and ridge regression both give almost similar results, the reason being both are used to curb the overfitting and make the model less complex.

After applying all the statistical learning techniques to build the machine learning model, we conclude that among all the regression techniques, polynomial regression with degree 1 gives the most accurate results on splitting the dataset into training and testing with the ratio of 80:20. It gives an accuracy of 33%. Our model gives a lesser accuracy because the dataset that we have chosen has very large-scale values of the parameter population and so it does not give very accurate results. On applying the ridge and lasso regression techniques and curbing the overfitting, the model finally gives an accuracy of 39%.

## Authors

- Aryaman : [@Aryaman-Arya](https://github.com/Aryaman-Arya)
- Sanjam Kaur Bedi : [@Sanjam-Bedi](https://github.com/Sanjam-Bedi)
