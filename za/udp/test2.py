#%%
import socket
import time
#创建socket对象
#SOCK_DGRAM  udp模式
s1=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s1.bind(("192.168.1.23",6000))
s2=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s2.bind(("192.168.1.23",6001))

# %%
for i in range(500):
    d = (str(i)+'aaaaa').encode()
    d2 = (str(i)+'bbbbb').encode()
    print(d)
    s1.sendto(d,("192.168.1.29",6000))
    s2.sendto(d2,("192.168.1.29",6001))
    data=s1.recv(1024) #一次接收1024字节
    if d != data:
        print(str(data),'  ',str(i))
        print('NG')
        break
    time.sleep(0.1)

# %%


#%%
data=s2.recv(1024)
print(data)
for i in range(500):
    d = (str(i)+'aaaaa').encode()
    d2 = (str(i)+'bbbbb').encode()
    print(d,' ',d2)
    s1.sendto(d,("192.168.1.29",6000))
    s2.sendto(d2,("192.168.1.29",6001))
    data=s1.recv(1024) #一次接收1024字节
    data2=s2.recv(1024) #一次接收1024字节
    if d != data or d2 != data2:
        print(str(data),'  ',str(data2),'  ',str(i))
        print('NG')
        break
    time.sleep(0.1)
#%%
while True:
    s1.sendto(('123'+'aaaaa').encode(),("192.168.1.29",6000))
    s2.sendto(('123'+'bbbbb').encode(),("192.168.1.29",6001))
    time.sleep(0.1)
# %%
