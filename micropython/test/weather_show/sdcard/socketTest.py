# import usocket
# import os 
# import time

# client = usocket.socket(usocket.AF_INET, usocket.SOCK_STREAM)
# client.connect(("192.168.1.12", 6000))
# for i in range(3):
#     client.send("rt-thread micropython!")
#     time.sleep(0.3)
# client.close()


import socket
import os 
import time

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("192.168.1.12", 6000))
for i in range(3):
    client.send("rt-thread micropython!")
    time.sleep(0.3)
client.close()
