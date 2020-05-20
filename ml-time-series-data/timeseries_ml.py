from glob import glob
import librosa as lr
import pandas as pd
import numpy as np

files = glob('data/heartbeat-sounds/files/*.wav')
print(files)

audio, sfreq = lr.load('data/heartbeat-sounds/proc/files/murmur_201101051104.wav')

indices = np.arange(0, len(audio))
time = indices / sfreq

final_time = (len(audio) - 1) / sfreq
time = np.linspace(0, final_time, sfreq)

df['date'] = pd.to_datetime(df['date'])

import librosa as lr
from glob import glob

# List all the wav files in the folder
audio_files = glob(data_dir + '/*.wav')

# Read in the first audio file, create the time array
audio, sfreq = lr.load(audio_files[0])
time = np.arange(0, len(audio)) / sfreq

# Plot audio over time
fig, ax = plt.subplots()
ax.plot(time, audio)
ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
plt.show()

# Read in the data
data = pd.read_csv('prices.csv', index_col=0)

# Convert the index of the DataFrame to datetime
data.index = pd.to_datetime(data.index)
print(data.head())

# Loop through each column, plot its values over time
fig, ax = plt.subplots()
for column in data.columns:
    data[column].plot(ax=ax, label=column)
ax.legend()
plt.show()