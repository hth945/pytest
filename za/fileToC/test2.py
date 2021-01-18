#%%
import os




out_file = open('test.txt', 'w')

for i in range(410):
    out_file.write('0123456789')
out_file.close()
print('complete')
# %%
out_file = open(r'D:\sysDef\download\test (4).txt', 'r')

for i in range(500000):
    strr = out_file.read(10)
    if strr != '0123456789':
        print(type(strr))
        print('err:',i)
        print(strr)
        
        break
out_file.close()
print('complete')
# %%

# %%
