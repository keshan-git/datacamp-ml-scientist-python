import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
data.head()

fig, ax = plt.subplots( figsize=(12, 6))
data.plot('data', 'close', ax=ax)
ax.set(title='AAPL daily closing price')

# Plot the time series in each dataset
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
data.iloc[:1000].plot(x='time', y='data_values', ax=axs[0])
data2.iloc[:1000].plot(x='time', y='data_values', ax=axs[1])
plt.show()

from sklearn.svm import LinearSVC

array.reshape([-1, 1]).shape

model = LinearSVC()
model.fit(X, y)
model.coef_

prediction = model.predict(X)

from sklearn.svm import LinearSVC

# Construct data for the model
X = data[['petal length (cm)', 'petal width (cm)']]
y = data[['target']]

# Fit the model
model = LinearSVC()
model.fit(X, y)

# Create input array
X_predict = targets[['petal length (cm)', 'petal width (cm)']]

# Predict with the model
predictions = model.predict(X_predict)
print(predictions)

# Visualize predictions and actual values
plt.scatter(X_predict['petal length (cm)'], X_predict['petal width (cm)'],
            c=predictions, cmap=plt.cm.coolwarm)
plt.title("Predicted class values")
plt.show()

# -------------------

from sklearn import linear_model

# Prepare input and output DataFrames
X = boston['AGE'].reshape(-1, 1)
y = boston['RM'].reshape(-1, 1)

# Fit the model
model = linear_model.LinearRegression()
model.fit(X, y)

# Generate predictions with the model using those inputs
predictions = model.predict(new_inputs.reshape(-1, 1))

# Visualize the inputs and predicted values
plt.scatter(new_inputs, predictions, color='r', s=3)
plt.xlabel('inputs')
plt.ylabel('predictions')
plt.show()

