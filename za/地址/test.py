#%%
a = 1
print(id(a))
for i in range(10):
    a = float(a+11)
    print(id(a))
    a = int(a+1)
    print(id(a))
# %%
