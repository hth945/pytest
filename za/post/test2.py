#%%
import requests
# url = "http://127.0.0.1"
url = "http://192.168.1.10"
# data = bytes("connection\r\n")
data = "connection\r\n"
res = requests.post(url=url,
                    data=data,
                    headers={'Content-Type': 'text/plain'})


print(res.text)
# %%


import requests
res = requests.post(url="http://192.168.1.5/",data="""C=ADD_RECORD&product=X1521&sn=PPPYWWDSSSSEEEERX+FFGGCUUCLPPPVHSSS&station_name=D-INSPECTION&station_id=SPWX_W03-2FT-01_1_D-INSPECTION&start_time=2019-06-28 08:11:28&stop_time=2019-06-28 08:11:39&result=PASS&reason=&stage=1&mac_address=88:51:FB:42:A1:35&value1=0.052&value2=0.052&value3=0.052&value4=0.052&value5=0.052&value6=0.052&value7=0.052&value8=0.052
""")
print(res.text)