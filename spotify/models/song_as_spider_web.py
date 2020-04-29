# Commented out IPython magic to ensure Python compatibility.
# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import extract_song_x as esx

song_x = esx.one_random_song_from_csv()

# prepare data for polar plot
feature_names = song_x.columns.str.title().values.tolist()
N = len(feature_names)

angles = [n / float(N) * 2 * np.pi for n in range(N)]

angles = np.hstack((angles, angles[0]))

values = song_x.values.flatten()
values = np.hstack((values, values[0]))

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='polar')

ax.set_title('Song features', weight='bold', size=20, position=(0.5, 1.1),
                 horizontalalignment='center', verticalalignment='center')

# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], feature_names, color='k', size=12, alpha=0.6)

# Draw rlabels
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2","0.4","0.6", "0.8"], color="k", size=10)
plt.ylim(0, 1)

# Plot data
ax.plot(angles, values,  linewidth=1, linestyle='solid')
ax.fill(angles, values, facecolor='g', alpha=0.1, closed=True)

# save fig
filename = '../../data/sample_feature_spiderweb.png'
plt.savefig(filename, dpi=600, facecolor='w', edgecolor='w')
