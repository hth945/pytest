#%%



# %%
#  服务器
# import socket  # 导入socket库
# import threading


# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
# s.bind(('192.168.1.200', 37075))             #绑定监听地址和端口
# s.listen(10)                             #指定等待连接的最大数量  


# # 处理tcp连接
# def tcplink(conn, addr):  
#     print("Accept new connection from %s:%s" % addr)  
#     # 向客户端发送欢迎消息  
#     conn.send(b"Server: Welcome!\n")  
#     while True:    
#         conn.send(b"Server: What's your name?")    
#         data = conn.recv(1024)    
#         # 如果客户端发送 exit 过来请求退出，结束循环    
#         if data == b"exit":      
#             conn.send(b"Server: Good bye!\n")      
#             break    
#         conn.send(b"Server: Hello %s!\n" % data)  
#         # 关闭连接  
#     conn.close()  
#     print("Connection from %s:%s is closed" % addr)



# while True:         
#     sock,addr=s.accept()                            #接受一个新连接 
#     t=threading.Thread(target=tcplink,args=(sock,addr)) #创建新线程来处理TCP连接 
#     t.start()                                       

# # %%

# #整合
# import socket  # 导入socket库
# import threading
# import time 

# # #服务器端        只开客户端的时候记得屏蔽
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
# s.bind(('192.168.1.200', 37075))             #绑定监听地址和端口
# s.listen(10)                             #指定等待连接的最大数量  

# # 处理tcp连接
# def tcplink(conn, addr):  
#     print("Accept new connection from %s:%s" % addr)  
#     # 向客户端发送欢迎消息  
#     conn.send(b"Server: Welcome!\n")  
#     while True:    
#         conn.send(b"Server: What's your name?")    
#         time.sleep(0.01)
#         # data = conn.recv(1024)    
#         # # 如果客户端发送 exit 过来请求退出，结束循环    
#         # if data == b"exit":      
#         #     conn.send(b"Server: Good bye!\n")      
#         #     break    
#         # conn.send(b"Server: Hello %s!\n" % data)  
#         # 关闭连接  
#     conn.close()  
#     print("Connection from %s:%s is closed" % addr)


# #客户端
# def TCP_client():
#     # 建立连接
#     while True:
#         client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         client.connect(("192.168.1.123", 8080))
#         client.send(b'Hello, Mr Right!\r\n') 
#         time.sleep(0.001)
#         # recv_data = client.recv(1024) 
#         # print("接收到的数据：%s" % recv_data)
#         # time.sleep(0.001)
#         client.close() 
#         time.sleep(0.01)

# # def TCP_client():
# #     # 建立连接
# #     while True:
# #         client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# #         client.connect(("192.168.1.123", 8080))
# #         client.send(b'Hello, Mr Right!\n') 
# #         time.sleep(0.001)
# #         # recv_data = client.recv(1024) 
# #         # print("接收到的数据：%s" % recv_data)
# #         # time.sleep(0.001)
# #         client.close() 
# #         time.sleep(0.05)
# if __name__ == "__main__":
#     sock,addr=s.accept()                            #接受一个新连接 
#     server_=threading.Thread(target=tcplink,args=(sock,addr))  #创建新线程来处理TCP连接 
#     server_.start()
#     TCP_client()
                                     



# %%
#  客户端 

import socket  # 导入socket库
import time 
# 创建一个socket
# 创建 socket 时，第一个参数 socket.AF_INET 表示指定使用 IPv4 协议，
# 如果要使用 IPv6 协议，就指定为 socket.AF_INET6。
# SOCK_STREAM 指定使用面向流的 TCP 协议。然后我们调用 connect() 方法，传入 IP 地址(或者域名)，指定端口号就可以建立连接了。



test_list = []

for j in range(0,16):
    for i in range(0,256):
        test_list.append(i)
# 建立连接

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("192.168.1.30", 80))
print('connect')
k=0
for i in range(10):
    s.send(bytes(test_list))
    print('send',len(bytes(test_list)))
    #time.sleep(0.5)
    recv_data = s.recv(len(test_list))
    print('recv: ',len(recv_data))

    for i in range(len(recv_data)):
        if test_list[(i+k)%4096] != bytes(recv_data)[i]:
            print(i,test_list[i],recv_data[i])
    k += len(recv_data)
    # print("接收到的数据：%s" % recv_data)
    time.sleep(0.005)
s.close() 
# %%
