#%%
from socket import *
 
serverSocket = socket(AF_INET,SOCK_STREAM)
 
serverSocket.bind(("",8000))
serverSocket.listen(5)
print("----------1----------")
clientSocket,clientInfo = serverSocket.accept() # 此处链接成功后才会输出2
print("-------2------------")

with open('123.txt','wb') as f:
    recvData = clientSocket.recv(1024) # 此处收到信息后才会输出3
    clientSocket.settimeout(1.0)
    print("---------3--------------")
    while True:
        print(recvData)
        f.write(recvData) 
        try:
            recvData = clientSocket.recv(1024) # 此处收到信息后才会输出3
        except:
            break

clientSocket.close()

# %%

