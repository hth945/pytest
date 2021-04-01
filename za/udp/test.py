#%%
#不需要建立连接
import socket
import time
#创建socket对象
#SOCK_DGRAM  udp模式
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

# s.setblocking(0) #非阻塞模式

# s.setblocking(1) #阻塞模式


s.bind(("192.168.1.23",6000))
#发送数据 字节
# s.sendto("1".encode(),("192.168.1.29",6000))
# data=s.recv(1024) #一次接收1024字节
# print(data.decode())


# %%
for i in range(500):
    d = (str(i)+'aaaaa').encode()
    print(d)
    s.sendto(d,("192.168.1.29",6000))
    print(i)
    data=s.recv(1024) #一次接收1024字节
    if d != data:
        print(str(data),'  ',str(i))
        print('NG')
        break
    time.sleep(0.1)


   # print(data.decode())
# %%
print(s.recv(1024))
time.sleep(3)
for i in range(500):
    d = (str(i)+'aaaaa').encode()
    print(d)
    s.sendto(d,("192.168.1.29",6000))
    print(i)
    data=s.recv(1024) #一次接收1024字节
    if d != data:
        print(str(data),'  ',str(i))
        print('NG')
        break
    time.sleep(0.1)
# %%
ord('H')/16
# %%
print('%x'%'HP'.encode()[0])
# %%
d=[0x3c,0x0e,0x04]
str(data)