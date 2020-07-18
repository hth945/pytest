#%%
import numpy as np
import matplotlib.pyplot as plt

speed = 400.0  #脉冲每秒  速度最快为100k  最小 20
Acceleration = 1000.0
Deceleration = 2000.0
X =1000.0 # 位移

Atime = speed/Acceleration
Dtime = speed/Deceleration
Ax = Atime*speed /2 
Dx = Dtime*speed /2 
if (Ax+Dx) > X: # 加不到最高速度
    T = np.sqrt(2*X/Acceleration+2*X/Deceleration)
    print('T: ',T)
    Atime = T*Deceleration/(Acceleration+Deceleration)
    Dtime = T - Atime
    Ax = Acceleration * Atime*Atime /2 
    Dx = Deceleration * Dtime*Dtime /2 

UniformTime = (X - (Ax + Dx))/speed
Ux = X - Ax - Dx
print(Atime)
print(UniformTime)
print(Dtime)
print(Ax)
print(Dx)
print(Ux)
    


#%%
time = np.linspace(0, 1, 2000) # s
n = int((Atime+ UniformTime+ Dtime)/0.001)
v = np.zeros([n], dtype=np.float32) # 
lv = 0
for i in range(n):
    if i < (Atime*1000):
        lv += Acceleration * 0.001
        v[i] = lv
    elif i >= ((Atime+UniformTime)*1000) and i < ((Atime+UniformTime+Dtime)*1000):
        lv -= Deceleration * 0.001
    v[i] = lv
plt.plot(v)
plt.show()

#%%
Ax
i = 1.0
n = 0
lT = 0
a = []
lT = int(np.sqrt(2*0/Acceleration)*1000000)
while i < Ax+1:
    T = int(np.sqrt(2*i/Acceleration)*1000000)
    # print(T - lT)
    a.append(T - lT)
    lT = T
    i += 1
    n += 1
i = 0.0
vTem = int(1000000/speed)
while i < Ux:
    a.append(vTem)
    i += 1
    n += 1
lT = int(np.sqrt(2*(Dx)/Deceleration)*1000000)
i = 1.0
while i < Dx+1:
    T = int(np.sqrt(2*(Dx-i)/Deceleration)*1000000)
    # print(lT - T)
    a.append(lT - T)
    n += 1
    lT = T
    i += 1

# %%
plt.plot(a)
plt.show()

# %%
int(1000000/speed)

# %%
n

# %%
a[0] /1000000.0

# %%
len(a)

# %%
