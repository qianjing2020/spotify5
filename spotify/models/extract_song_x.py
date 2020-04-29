# Commented out IPython magic to ensure Python compatibility.
# imports
from sklearn import preprocessing
import os
import numpy as np
import pandas as pd

def one_random_song_from_csv():
    df = pd.read_csv(
    '/Users/jing/Documents/LambdaSchool/spotify_app/data/spotify2019.csv')

    # normalize song features
    new = df.select_dtypes(include=np.number)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(new)
    normalized = pd.DataFrame(x_scaled)
  
    # create new dataframe store normalized song features
    song_features = df.select_dtypes(include=np.number).columns.tolist()
    old_colnames = normalized.columns.tolist()

    x = dict(zip(old_colnames, song_features))
    df2 = normalized.rename(columns=x, index=df['track_id'])
    
    song_x = df2.sample(1)
    return song_x
