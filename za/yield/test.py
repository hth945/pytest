#%%

import sys
import time

def fibonacci(n): # 生成器函数 - 斐波那契
    a, b, counter = 0, 1, 0
    while True:
        if (counter > n): 
            return
        print("yield1: ",a)
        yield(a)
        print("yield2: ",a)
        time.sleep(1)
        print("yield3: ",a)
        a, b = b, a + b
        counter += 1

f = fibonacci(3) # f 是一个迭代器，由生成器返回生成
 
while True:
    try:
        time.sleep(1)
        print("o:")
        print(next(f))
    except StopIteration:
        sys.exit()
#%%
for i in fibonacci(10):
    print(i)
    time.sleep(1)

# %%
