import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
import numpy as np
import extract_song_x as esx
import radarplot


# prepare data for radar plot
song_x = esx.one_random_song_from_csv()
spoke_labels = song_x.columns.tolist()
N = len(spoke_labels)

theta = radarplot.radar_factory(N, frame='polygon')

data = song_x.values.flatten()

color_list = list(colors._colors_full_map.values())
random.seed(9)
colors = random.sample(color_list, 14)

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
ax = plt.gca()
ax.set_title('Song features', weight='bold', size=20, position=(0.5, 1.1),
             horizontalalignment='center', verticalalignment='center')

for i in range(0, 14):
    xx = [0, theta[i]]
    yy = [0, data[i]]
    ax.plot(xx, yy, marker='s', markersize=10, linewidth=10, color=colors[i])
    ax.fill(xx, yy, facecolor=colors[i], alpha=0.25, closed=True)
ax.set_varlabels(spoke_labels)

filename = '../../data/sample_feature_urchin.png'
plt.savefig(filename, dpi=600, facecolor='w', edgecolor='w')
