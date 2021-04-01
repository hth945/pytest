#%%
from socket import *
 
serverSocket = socket(AF_INET,SOCK_STREAM)
 
serverSocket.bind(("",8000))
serverSocket.listen(5)
print("开始监听")

while True:
    clientSocket,clientInfo = serverSocket.accept() # 此处链接成功后才会输出2

    print('连接 ', clientInfo)
    while True:
        try:
            recvData = clientSocket.recv(1024) # 此处收到信息后才会输出3
            if len(recvData) == 0:
                break
            print(recvData)
        except:
            break
    print('连接断开')
    clientSocket.close()
# %%
