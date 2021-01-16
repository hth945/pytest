#%%

b1 = bytes()
b1[0]=0
print(b1)

# %%
a=b'\x30\x00\x31'
print(a)
# %%
str1 = a.decode('ASCII')
print("str1: ", str1)
# %%
a=b'\x30\x00\x31'
print(a)
str1 = a.decode('ASCII')
print("str1: ", str1)
b=bytes(str1,encoding='ASCII')
print(list(b))
# %%
