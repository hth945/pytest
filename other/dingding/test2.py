#%%
# -*- coding: utf-8 -*-
import requests

r = requests.get("https://oapi.dingtalk.com/gettoken?appkey=dingqipjrmzccr6suavo&appsecret=oMhvkVfnSRPP-E4G5fYzx7fs-pSyikehZoxEvtRn_Bgm1AyfFznIYtjygJIUw0tT")

print(r.text)
# %%
import requests

r = requests.get("https://oapi.dingtalk.com/user/getuserinfo?access_token=5ebed1dd92a935d4a36f18b2aa5702ab")

print(r.text)
# %%
