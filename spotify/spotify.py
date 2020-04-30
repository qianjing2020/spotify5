# imports 
#from __future__ import print_function  # (at top of module)
from dotenv import load_dotenv, find_dotenv
import os
import sys
import time
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from pprint import pprint

# setup spotify api
cwd = os.getcwd()
sys.path.insert(0, cwd)
load_dotenv(find_dotenv())

cid = os.getenv(spotify_id)
secret = os.getenv(spotify_token)

client_credentials_manager = SpotifyClientCredentials(spotify_id, secret)

sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

search_str = 'Muse'
results = sp.search(search_str, limit=20)

for idx, track in enumerate(results['track']['time']):
    print(idx, track['name'])
    
