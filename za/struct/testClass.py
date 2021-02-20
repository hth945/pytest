#%%

import struct

class test:
    def __init__(self,):
        a=bytes('hello ',encoding="ascii")
        b=bytes('world!',encoding="ascii")
        c=2
        d=45.123
        self.structBytes=bytearray(struct.pack('4s5sbf',a,b,c,d))

    @property
    def a(self):
        return struct.unpack("4s", self.structBytes[0:4])[0]

    @a.setter
    def a(self, a):
        tem = bytes(a,encoding="ascii")
        self.structBytes[0:4]=bytearray(struct.pack('4s',tem))

t = test()
print(t.a)
t.a='456789'
print(t.a)
        



# %%
