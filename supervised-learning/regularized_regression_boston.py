import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

boston = pd.read_csv('data/boston.csv', index_col='id')
print(boston.columns)
print(boston.head())

X = boston.drop('medv', axis=1).values
y = boston['medv'].values
names = boston.drop('medv', axis=1).columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
score = ridge.score(X_test, y_test)

print(score)

lasso = Lasso(alpha=0.1, normalize=True)
lasso_fit=lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
score = lasso.score(X_test, y_test)

print(score)

_ = plt.plot(range(len(names)), lasso_fit.coef_)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()