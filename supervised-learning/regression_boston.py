import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

boston = pd.read_csv('data/boston.csv', index_col='id')
print(boston.columns)
print(boston.head())

X = boston.drop('medv', axis=1).values
y = boston['medv'].values

X_rooms = X[:, 5].reshape(-1, 1)
y = y.reshape(-1, 1)

plt.scatter(X_rooms, y)
plt.show()

sns.heatmap(boston.corr(), square=True, cmap='RdYlGn')

reg = LinearRegression()
reg.fit(X_rooms, y)

predication_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1, 1)
plt.scatter(predication_space, reg.predict(predication_space), color='black', linewidth=3)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)

print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

reg = LinearRegression()
cv_result = cross_val_score(reg, X, y, cv=5)

print(cv_result)
print(np.mean(cv_result))
