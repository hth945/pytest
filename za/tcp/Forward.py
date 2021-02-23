#%%

from socket import *
import _thread
import threading

fileWriteFlag = -1
lock=threading.Lock()
# with open('forward.txt','wb') as f:

def connectServer():
    global ser
    global cli
    global fileWriteFlag

    try:
        while True:
            b = ser.recv(1024) # 被关闭
            if len(b) == 0:
                break
            lock.acquire()
            with open('forward.txt','ab') as f:
                if fileWriteFlag != 0:
                    fileWriteFlag=0
                    f.write(bytes("\n---------ser.rec--------\n",encoding='ascii'))
                f.write(b)
            lock.release()
            print('ser.recv:',b[0])
            cli.send(b)
    except Exception as e:
        print('错误明细是',e)
    print('ser.exit()')
    ser.close()
    cli.close()

global fileWriteFlag
cliSer = socket(AF_INET,SOCK_STREAM)
cliSer.bind(("",7999))
cliSer.listen(5)
while True:
    cli,clientInfo = cliSer.accept() # 此处链接成功后才会输出2

    ser = socket(AF_INET, SOCK_STREAM)
    ser.connect(("127.0.0.1", 8000))

    _thread.start_new_thread(connectServer,())
    try:
        while True:
            b = cli.recv(1024)
            if len(b) == 0:
                break
            lock.acquire()
            with open('forward.txt','ab') as f:
                if fileWriteFlag != 1:
                    fileWriteFlag=1
                    f.write(bytes("\n---------cli.rec--------\n",encoding='ascii'))
                f.write(b)
            lock.release()
            print('cli.recv:',b)
            ser.send(b)
    except Exception as e:
        print('错误明细是',e)
    print('cli.exit()')
    ser.close()
    cli.close()

# %%
with open('forward.txt','wb') as f:
    tem = bytes("\n------------\n",encoding='ascii')
    f.write(tem)

with open('forward.txt','ab') as f:
    tem = bytes("\n------------\n",encoding='ascii')
    f.write(tem)
# %%
