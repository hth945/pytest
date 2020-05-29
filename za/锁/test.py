#%%
import time
import _thread
import threading

count=0 #声明全局变量
lock=threading.Lock() #申请一把锁
def test():
    global count
    # lock.acquire()
    a = count
    time.sleep(1)
    a += 1
    count = a
    print(count)
    # lock.release()
    time.sleep(1)


def run():
    while True:
        test()



th = _thread.start_new_thread(run,()) #声明线程数

run()



# %%
