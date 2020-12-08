#%%
import serial



ser=serial.Serial("COM6",9600,timeout=0.5)#winsows系统使用com1口连接串行口
# ser=serial.Serial("/dev/ttyS1",9600,timeout=0.5)#Linux系统使用com1口连接串行口


ser.write(bytes("123", encoding="utf8"))#向端口些数据
s = ser.read(10)#从端口读10个字节
print(s)

s = ser.readline()#从端口读10个字节
if len(s)==0:
    print('kkk')
print(s)
print(str(s[:-1], encoding="utf-8"))
ser.close()#关闭端口

# %%
s
# %%

# %%
