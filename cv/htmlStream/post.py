#%%
import requests
url = "http://127.0.0.1:5000/video_feed"
data = {"key":"value"}
res = requests.get(url=url)


# %%
res.text