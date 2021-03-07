#%%

from socket import *

ser = socket(AF_INET, SOCK_STREAM)
ser.connect(("3p586243x1.zicp.vip", 13546))
ser.send(b"Server: Welcome!\n") 
ser.close()
# while True:

#     b = ser.recv(1024)
#     print('ser.recv b:',len(b))
#     print('ser.recv b:',b)
#     print('ser.recv b[0]:',b[0])
#     print(b)
# %%
ser.close()
# %%

# strd = b"""GET / HTTP/1.1
# Host: 127.0.0.1
# Connection: keep-alive
# Cache-Control: max-age=0
# sec-ch-ua: "Chromium";v="88", "Google Chrome";v="88", ";Not A Brand";v="99"
# sec-ch-ua-mobile: ?0
# Upgrade-Insecure-Requests: 1
# User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36
# Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9
# Sec-Fetch-Site: none
# Sec-Fetch-Mode: navigate
# Sec-Fetch-User: ?1
# Sec-Fetch-Dest: document
# Accept-Encoding: gzip, deflate, br
# Accept-Language: zh-CN,zh;q=0.9
# If-Modified-Since: Sat, 06 Feb 2021 06:04:55 GMT

# """
strd = b"""GET / HTTP/1.1
"""
from socket import *

ser = socket(AF_INET, SOCK_STREAM)
ser.connect(("3p586243x1.zicp.vip", 13546))
ser.send(strd) 
ser.close()
# %%
