import numpy as np

fig, axs = plt.subplots(1, 2)

axs[0].plot(x, c='k', lw=3, alpha=0.2)
axs[0].ploy(y)
axs[0].set(xlabel='time', title='X values = time')

axs[1].scatter(x_long, y_long, c=np.arange(len(x_long)), camp='viridis')
axs[1].set(xlabel='x', ylabel='y', title='Color = time')


# Plot the raw values over time
prices.plot()
plt.show()

# Scatterplot with one company per axis
prices.plot.scatter('EBAY', 'YHOO')
plt.show()

# Scatterplot with color relating to time
prices.plot.scatter('EBAY', 'YHOO', c=prices.index,
                    cmap=plt.cm.viridis, colorbar=False)
plt.show()