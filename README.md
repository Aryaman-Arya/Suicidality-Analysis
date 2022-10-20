# Suicidality Analysis for assorted demographic Mental Health Records

Analysed the risk of suicides for varying compositions of the human population by implementing possible regression models to get the most accurate model on the basis of their Mental Health Records.

## Tech Stack

**[Python 3.10](https://www.python.org/downloads/release/python-3106/)** 

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

##Plot 1: HEATMAP IMAGE

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
##Plot 2

#### With Split

**Taking 70% Training and 30% Testing data**

```bash
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state=0)
```

##Plot 3

**Taking 80% Training and 20% Testing data**

```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=0)
```
##Plot 3

**Taking 50% Training and 50% Testing data**

```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=0)
```
##Plot 3


## Authors

- Aryaman : [@Aryaman-Arya](https://github.com/Aryaman-Arya)
- Sanjam Kaur Bedi : [@Sanjam-Bedi](https://github.com/Sanjam-Bedi)
