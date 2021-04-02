#%%

import dingtalk.api

# request = dingtalk.api.OapiMediaUploadRequest("https://oapi.dingtalk.com/media/upload")
# request.type = "image"
# request.media = dingtalk.api.FileItem("/Users/zero/test.png",open("/Users/zero/test.png","rb"))
# resp = request.getResponse("*****************")
# print(resp)


request = dingtalk.api.OapiGettokenRequest("https://oapi.dingtalk.com/gettoken")
request.corpid="***"
request.corpsecret="**-***"

f = request.getResponse()
print(f)

request = dingtalk.api.OapiXiaoxuanPreTest1Request("https://oapi.dingtalk.com/topapi/xiaoxuan/pre/test1")
request.normalData="1"
request.systemData="2"

f = request.getResponse("***************")
print(f)
    
request = dingtalk.api.OapiServiceGetAuthInfoRequest("https://oapi.dingtalk.com/service/get_auth_info")
request.auth_corpid = "******************"
f = request.getResponse(accessKey="******",
    accessSecret="**************",
    suiteTicket="***********")
print(f)
# %%
import time
import hmac
import base64
from hashlib import sha256
import urllib
import json
import requests

    #获取code
code = 'beb21642e4503abeb079cccc4f06420a'

t = time.time()
#时间戳
timestamp = str((int(round(t * 1000))))
appSecret ='FNFTskwdqNpjoX6i8iq9rr0Uv9flMYSKaLgR23xarSrG1Ex3D5YL8hPPtvZh25cJ'
AppId = 'dingoa6jgust3skqrq9hak'
#构造签名
signature = base64.b64encode(hmac.new(appSecret.encode('utf-8'),timestamp.encode('utf-8'), digestmod=sha256).digest())
#请求接口，换取钉钉用户名
payload = {'tmp_auth_code':code}
headers = {'Content-Type': 'application/json'}
res = requests.post('https://oapi.dingtalk.com/sns/getuserinfo_bycode?signature='+urllib.parse.quote(signature.decode("utf-8"))+"&timestamp="+timestamp+"&accessKey="+AppId,data=json.dumps(payload),headers=headers)

res_dict = json.loads(res.text)
print(res_dict)

# %%
