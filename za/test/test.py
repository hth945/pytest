#%%
a=0
t = 0
for block_count in range(1,70):
    if block_count % 4 == 0  and block_count/4%7 != 0:
        a += 42
        t += 1
    else:
        a += 41
a += 36
print(a)
# %%
a/3*2
# %%
1920-36
# %%
