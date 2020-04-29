# imports 
from __future__ import print_function  # (at top of module)
import os
import sys
import time

import json

from .. import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


# setup spotify api
cwd = os.getcwd()
sys.path.insert(0, cwd)

client_credentials_manager = SpotifyClientCredentials(
    client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

#print(dir(sp))
"""

if len(sys.argv) > 1:
    tid = sys.argv[1]
else:
    tid = 'spotify:track:4TTV7EcfroSLWzXRY6gLv6'

start = time.time()
analysis = sp.audio_analysis(tid)
delta = time.time() - start
print(json.dumps(analysis, indent=4))
print("analysis retrieved in %.2f seconds" % (delta,))


# get data
# create empty lists where the results are going to be stored
start = time.process_time()

artist_name = []
track_name = []
popularity = []
track_id = []

# for i in range(0, 10, 1):
#     track_results = sp.search(q='year:2020', type='track', limit=50, offset=i)
#     print(dir(track_results))
#     # for i, t in enumerate(track_results['tracks']['items']):
    #     artist_name.append(t['artists'][0]['name'])
    #     track_name.append(t['name'])
    #     track_id.append(t['id'])
    #     popularity.append(t['popularity'])
search_str = 'Radiohead'
result = sp.search(search_str)

end = time.process_time
print(f'process time in seconds: {end-start}')
"""
