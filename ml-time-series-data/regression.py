from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

alphas = [0.1, 1e2, 1e3]
ax.plot(y_test, color='k', alpha=0.3, lw=3)

for ii, alpha in enumerate(alphas):
    y_predicted = Ridge(alpha=alpha).fit(X_train, y_ytrain).predict(X_test)
    ax.plot(y_predicted, c=cmap(ii / length(alphas)))

ax.legend(['True values', 'Model 1', 'Model 2', 'Model 3'])
ax.set(xlabel='Time')

score = r2_score(y_predicted, y_test)

#--
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Use stock symbols to extract training data
X = all_prices[['EBAY', 'NVDA', 'YHOO']]
y = all_prices[['AAPL']]

# Fit and score the model with cross-validation
scores = cross_val_score(Ridge(), X, y, cv=3)
print(scores)

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Split our data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=.8, shuffle=False, random_state=1)

# Fit our model and generate predictions
model = Ridge()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
print(score)

# Visualize our predictions along with the "true" values, and print the score
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(y_test, color='k', lw=3)
ax.plot(predictions, color='r', lw=2)
plt.show()