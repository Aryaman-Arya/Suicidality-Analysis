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


## Authors

- Aryaman : [@Aryaman-Arya](https://github.com/Aryaman-Arya)
- Sanjam Kaur Bedi : [@Sanjam-Bedi](https://github.com/Sanjam-Bedi)
