# in progress...

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv('/Users/jing/Documents/LambdaSchool/spotify_app/data/spotify2019.csv')

df = df.set_index('track_id')
df = df.select_dtypes(include='number')
features = df.columns.drop(['energy'])
target = 'energy'

y = df[target]
x = df[features]

y = pd.qcut(y, np.linspace(0,1,num=7), labels = [i for i in range(0,6)])
standardScaler = preprocessing.StandardScaler()
x_normalized = standardScaler.fit_transform(x)

n_components = 9 # trial-proved
pca = PCA(n_components=n_components, random_state=42)  #how many compoment we want to retain
principle_components = pca.fit_transform(x_normalized)

# try kmean as baseline
# elbow plot find best number of clusters
ks = range(1, 7)
inertias = [] # sum squred distance to centroid
for k in ks:
    # Create a KMeans instance with k clusters: model
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(principle_components)
    inertias.append(kmeans.inertia_)
    
plt.plot(ks, inertias, '-ko')
plt.xlabel('k clusters')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# so let's use 6 clusters based on above plot
kmeans = KMeans(n_clusters=6)
kmeans.fit(principle_components)

y_pred = kmeans.labels_
centroids = kmeans.cluster_centers_

accuracy = accuracy_score(y, y_pred)
print(f'KMean accuracy is {accuracy}')
print('Low baseline! Try another algorithm!')

# Use KNN model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    principle_components, y.ravel(), test_size=0.2, random_state=42)

# find optimal neighbors
neighbors = np.arange(1, 8) 
train_accuracy = np.empty(len(neighbors)) 
test_accuracy = np.empty(len(neighbors)) 
  
# Loop over K values of neighbors
for i, k in enumerate(neighbors): 
    knn = KNeighborsClassifier(n_neighbors=k) 
    knn.fit(x_train, y_train) 
      
    # Compute traning and test data accuracy 
    train_accuracy[i] = knn.score(x_train, y_train) 
    test_accuracy[i] = knn.score(x_test, y_test) 

plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy') 
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy') 
  
plt.legend() 
plt.xlabel('n_neighbors') 
plt.ylabel('Accuracy') 
plt.show()

# based on above plot, select n_neighbor=5
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)
print(f'Train accuracy: {knn.score(x_train, y_train)}, test accuracy: {knn.score(x_test, y_test)}')

from sklearn.neighbors import kneighbors_graph
A = kneighbors_graph(x, 5, mode='distance', include_self=True, n_jobs=-1)
# A is a sparse matrix with A[i,j]=connectivity from i to j
# A !=0  Dont run this line!!!!!

from sklearn.neighbors import NearestNeighbors
song_x = df.sample(1)
song_x

song_x =standardScaler.transform(song_x)

# suggest 5 songs similar to song_x use euclidean distance
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(principle_components)
distances, indices = nbrs.kneighbors(x)
