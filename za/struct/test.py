#%%
from ctypes import *
class SSHead(Structure):
    _pack_ = 4
    _fields_ = [
        #(字段名, c类型 )
        ('nTotalSize', c_uint32),
        ('nSourceID', c_int32),
        ('sourceType', c_uint8),
        ('destType', c_uint8),
        ('transType', c_uint8),
        ('nDestID', c_int32),
        ('nFlag', c_uint8),
        ('nOptionalLength', c_uint16),
        ('arrOptional', c_char * 20),
    ]
    
    def encode(self):
        return string_at(addressof(self), sizeof(self))

    def decode(self, data):
        memmove(addressof(self), data, sizeof(self))
        return len(data)

# -------------------
# 使用
sshead = SSHead()
sshead.nSourceID = 20 #省略其他赋值
buf = sshead.encode()

ss = SSHead()
ss.decode(buf)
print(ss.nSourceID)
# %%

# %%
