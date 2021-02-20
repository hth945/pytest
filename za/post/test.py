#%%
import requests
# url = "http://127.0.0.1"
url = "http://192.168.1.10/runcmd"
# data = bytes("connection\r\n")
data = "connection\r\n"
res = requests.post(url=url,
                    data=data,
                    headers={'Content-Type': 'application/octet-stream'})


print(res.text)
# %%
import requests
res = requests.post(url="http://192.168.1.5/runcmd",data="connection\r\n")
print(res.text)
res = requests.post(url="http://192.168.1.5/runcmd",data="setLed:0\r\n")
# 500-2500
res = requests.post(url="http://192.168.1.5/runcmd",data="setD5:3000us\r\n")
res = requests.post(url="http://192.168.1.5/runcmd",data="setD6:2000us\r\n")
res = requests.post(url="http://192.168.1.5/runcmd",data="setD7:3000us\r\n")
res = requests.post(url="http://192.168.1.5/runcmd",data="setD8:3000us\r\n")
# %%
import requests
import time
for i in range(10):
    res = requests.post(url="http://192.168.1.5/runcmd",data="setD5:500us\r\n")
    time.sleep(0.5)
    res = requests.post(url="http://192.168.1.5/runcmd",data="setD5:2500us\r\n")
    time.sleep(0.5)
# %%
