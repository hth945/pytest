#%%

from socket import *

s = socket(AF_INET, SOCK_STREAM)
s.connect(("127.0.0.1", 8000))
print('connect')

with open('123.txt','rb') as f:

    s.send(f.read())

    s.settimeout(1.0)
    while True:
        try:
            recvData = s.recv(1024)
            print(recv_data)
        except:
            break
        
    s.close()
# %%
