feats = prices.rolling(20).aggregate([np.std, np.max]).dropna()
feats.head()

import numpy as np
from functools import partial

mean_over_first_axis = partial(np.mean, axis=0)

a = np.array([[1, 2], [1, 2]])
mean_over_first_axis(a)

# percentile
np.percentile(np.linspace(0, 200), q=20)


data = np.linspace(0, 100)

percentile_funcs = [partial(np.percentile, q=ii) for ii in [20, 40, 60]]
percentiles = [i_func(data) for i_func in percentile_funcs]
print(percentiles)

data.rolling(20).aggregate(percentiles)

# date-base
price.index = pd.to_datetime(price.index)
day_of_week_num = price.index.weekday

day_of_week = prices.index.weekday_name

# ---------------------------------------------
# Define a rolling window with Pandas, excluding the right-most datapoint of the window
prices_perc_rolling = prices_perc.rolling(20, min_periods=5, closed='right')

# Define the features you'll calculate for each window
features_to_calculate = [np.min, np.max, np.mean, np.std]

# Calculate these features for your rolling window object
features = prices_perc_rolling.aggregate(features_to_calculate)

# Plot the results
ax = features.loc[:"2011-01"].plot()
prices_perc.loc[:"2011-01"].plot(ax=ax, color='k', alpha=.2, lw=3)
ax.legend(loc=(1.01, .6))
plt.show()

# Import partial from functools
from functools import partial
percentiles = [1, 10, 25, 50, 75, 90, 99]

# Use a list comprehension to create a partial function for each quantile
percentile_functions = [partial(np.percentile, q=percentile) for percentile in percentiles]

# Calculate each of these quantiles on the data using a rolling window
prices_perc_rolling = prices_perc.rolling(20, min_periods=5, closed='right')
features_percentiles = prices_perc_rolling.aggregate([func for func in percentile_functions])

# Plot a subset of the result
ax = features_percentiles.loc[:"2011-01"].plot(cmap=plt.cm.viridis)
ax.legend(percentiles, loc=(1.01, .5))
plt.show()

# Extract date features from the data, add them as columns
prices_perc['day_of_week'] = prices_perc.index.weekday
prices_perc['week_of_year'] = prices_perc.index.weekofyear
prices_perc['month_of_year'] = prices_perc.index.month

# Print prices_perc
print(prices_perc)