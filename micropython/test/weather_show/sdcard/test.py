import os 
import time
from userfunc import *
# from skyFun import *
import skyFun
from sdcard.skyClass import * 

b = skyFun.getBinData('123')

print(type(b))
b[0]
print(len(b))
print(b)
b[0] = 1
b[4] = 2
b[16] = 2

from skyClass import * 

pbt = pyBinTest()
# pbt.a=1
pbt.b=2
pbt.c=bytearray((1,2))
pbt.d=1.1
pbt.e=bytearray((1,2,3))
print(pbt.a)
print(pbt.b)
print(pbt.c)
print(pbt.d)

if pbt.a < 3:
    print('1')
else:
    print('2')


from skyFun import *
MIPI_WR((0x39,0xFF,0x20))
MIPI_WR((0x39,0xFB,0x01))

MIPI_WR((0x39,0x89,0x33))


MIPI_WR((0x39,0x8A,0x33))
MIPI_WR((0x39,0x8B,0x33))
MIPI_WR((0x39,0x8C,0x33))


for i in range(3):
    MIPI_WR((0x39,0x89,0x43))
    time.sleep(1)
    MIPI_WR((0x39,0x89,0x33))
    time.sleep(1)



data = ((0x39,0x89,0x43),
(0x39,0x89,0x53),
(0x39,0x89,0x33),
)

for j in data:
    MIPI_WR(j)
    time.sleep(1)

# for i in range(10):
#     s = student('ken',10,60,3)
#     s.age=10
#     print(s.age)
#     # s.speak()


# led = skyFun.LED(4)
# print(led.TOP_LEFT_1)
# led.on()
# print(led.name)
# led = skyFun.LED(skyFun.LED.TOP_LEFT_1)
# led.on()



# my_function()
# print(add(666,111))
# print(add2(666,111))

# import userfunc

# userfunc.my_function()
# print(userfunc.add(666,111))


# print('234')
# print(os.listdir())
# for i in range(10):
#     time.sleep(1)
#     print(i)



