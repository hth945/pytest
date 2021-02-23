#%%

from socket import *

ser = socket(AF_INET, SOCK_STREAM)
ser.connect(("127.0.0.1", 6000))

while True:
    b = ser.recv(1024)
    print('ser.recv b:',len(b))
    print('ser.recv b:',b)
    print('ser.recv b[0]:',b[0])
    print(b)
# %%
