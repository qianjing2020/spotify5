
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

cid = os.getenv(spotify_id)
secret = os.getenv(spotify_token)


