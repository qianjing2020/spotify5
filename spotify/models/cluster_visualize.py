import plotly.graph_objects as go
from numpy import savetxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from screeplot import scree_plot
import plotly.express as px


# # setup chart studio api
# cwd = os.getcwd()
# sys.path.insert(0, cwd)
# load_dotenv(find_dotenv())

# cs_id = os.getenv(plotly_id)
# cs_api = os.getenv(plotly_api)

# chart_studio.tools.set_credentials_file(username = cs_id, api_key=cs_api)


# read data
df = pd.read_csv('/Users/jing/Documents/LambdaSchool/spotify_app/data/spotify2019.csv')

# set target for classification
df = df.set_index('track_id')
df = df.select_dtypes(include='number')
features = df.columns.drop(['energy'])
target = 'energy'
y = df[target]
x = df[features]
# convert target to label
y = pd.qcut(y, np.linspace(0,1,num=7), labels = [i for i in range(0,6)])

standardScaler = preprocessing.StandardScaler()
x_normalized = standardScaler.fit_transform(x)

n_components = 9 # trial-proved
pca = PCA(n_components=n_components, random_state=42)
principle_components = pca.fit_transform(x_normalized) # x_pc are principle components

#scree_plot(pca)

var_explained = pca.explained_variance_ratio_
print(f'variation explained by {n_components} components: {np.sum(var_explained)}')

# cluster plot along pc axes
y_plot = y.astype(int).values.reshape(-1, 1)

col_names =['PCA'+str(i) for i in range(0,9)]
col_names += 'y'
data = np.hstack((principle_components, y_plot))
pc_df = pd.DataFrame(data, columns=col_names)
pc_df.head(3)

# plot clusters
fig = plt.subplots(figsize=(10,9))
axes = plt.gca()

axes.scatter(pc_df['PCA0'],
             pc_df['PCA1'],  
             c = y,
             cmap = 'gist_rainbow', 
             alpha=0.1)

axes.set_xlabel("PCA1", fontsize=16)
axes.set_ylabel("PCA2", fontsize=16)
plt.title('Song clustering', fontsize=16)
plt.show()
filename = 'data/clustering.png'
plt.savefig(filename, dpi=600, facecolor='w', edgecolor='w')

# 3d plot 
data = pc_df
fig = px.scatter_3d(data, x='PCA0', y='PCA1', z='PCA2', color='y', opacity=0.2)
fig.update_traces(marker=dict(size=2))
fig.show()
fig.write_html("data/Interactive_3D_cluster.html")
