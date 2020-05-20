# Interpolation

missing = prices.isna()

prices_interp = prices.interpolate('linear')

ax = prices_interp.plot(c='r')
prices.plot(c='k', ax=ax, lw=2)


# rolling window
def percent_change(values):
    previous_values = values[:-1]
    last_value = values[-1]

    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return percent_change

fig, axs = plt.subplots(1, 2, figsize=(10,5))
ax = prices.plot(ax=axs[0])

ax = prices.rolling(window=20).aggregate(percent_change).plot(ax=axs[1])
ax.legend_.set_visible(False)

# outliers

fig, axs = plt.subplots(1,2, figsize=(10,5))

for data, ax in zip([prices, prices_perc_change], axs):
    t_mean = data.mean()
    t_std = data.std()

    data.plot(ax=ax)

    ax.axhline(t_mean + t_std * 3, ls='--', c='r')
    ax.axhline(t_mean - t_std * 3, ls='--', c='r')

prices_outlier_centered = prices_outlier_perc - prices_outlier_perc.mean()

std = prices_outlier_perc.std()

outliers = np.abs(prices_outlier_centered) > (std * 3)

prices_outlier_fixed = prices_outlier_centered.copy()
prices_outlier_fixed[outliers] = np.nanmedian(prices_outlier_fixed)

fig, axs = plt.subplots(1, 2, figsize=(10,5))
prices_outlier_centered.plot(ax=axs[0])
prices_outlier_fixed.plot(ax=axs[1])


# ------------------
# Visualize the dataset
prices.plot(legend=False)
plt.tight_layout()
plt.show()

# Count the missing values of each time series
missing_values = prices.isna().sum()
print(missing_values)


# Create a function we'll use to interpolate and plot
def interpolate_and_plot(prices, interpolation):
    # Create a boolean mask for missing values
    missing_values = prices.isna()

    # Interpolate the missing values
    prices_interp = prices.interpolate(interpolation)

    # Plot the results, highlighting the interpolated values in black
    fig, ax = plt.subplots(figsize=(10, 5))
    prices_interp.plot(color='k', alpha=.6, ax=ax, legend=False)

    # Now plot the interpolated values on top in red
    prices_interp[missing_values].plot(ax=ax, color='r', lw=3, legend=False)
    plt.show()


# Interpolate using the latest non-missing value
interpolation_type = 'zero'
interpolate_and_plot(prices, interpolation_type)

# Interpolate linearly
interpolation_type = 'linear'
interpolate_and_plot(prices, interpolation_type)

# Interpolate with a quadratic function
interpolation_type = 'quadratic'
interpolate_and_plot(prices, interpolation_type)

# Your custom function
def percent_change(series):
    # Collect all *but* the last value of this window, then the final value
    previous_values = series[:-1]
    last_value = series[-1]

    # Calculate the % difference between the last value and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return percent_change

# Apply your custom function and plot
prices_perc = prices.rolling(20).aggregate(percent_change)
prices_perc.loc["2014":"2015"].plot()
plt.show()


def replace_outliers(series):
    # Calculate the absolute difference of each timepoint from the series mean
    absolute_differences_from_mean = np.abs(series - np.mean(series))

    # Calculate a mask for the differences that are > 3 standard deviations from zero
    this_mask = absolute_differences_from_mean > (np.std(series) * 3)

    # Replace these values with the median accross the data
    series[this_mask] = np.nanmedian(series)
    return series


# Apply your preprocessing function to the timeseries and plot the results
prices_perc = prices_perc.aggregate(replace_outliers)
prices_perc.loc["2014":"2015"].plot()
plt.show()



