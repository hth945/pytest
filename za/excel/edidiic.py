#%%
from openpyxl import load_workbook

wb2 = load_workbook("edidIIC.xlsx")

print(wb2.sheetnames)
ws = wb2[wb2.sheetnames[0]]
# %%
row = 1
liftEdid1=""
liftEdid2=""
rightEdid1=""
rightEdid2=""
while True:
    d=ws.cell(row,1)
    if d.value==None:
        break
    # if int(d.value,16)  == 0x0227: # 左
    #     liftEdid1+= ' ' + hex(int(ws.cell(row,2).value,16))[2:]  
    #     liftEdid2+= ',' + hex(int(ws.cell(row,2).value,16))
    # elif int(d.value,16)  == 0x0C27: # 左
    #     rightEdid1+= ' ' + hex(int(ws.cell(row,2).value,16))[2:]
    #     rightEdid2+= ',' + hex(int(ws.cell(row,2).value,16))

    if int(d.value,16)  == 0x0227: # 左
        liftEdid1+= ' {:02x}'.format(int(ws.cell(row,2).value,16))
        liftEdid2+= '0x{:02x},'.format(int(ws.cell(row,2).value,16))
    elif int(d.value,16)  == 0x0C27: # 左
        rightEdid1+= ' {:02x}'.format(int(ws.cell(row,2).value,16))
        rightEdid2+= '0x{:02x},'.format(int(ws.cell(row,2).value,16))


    row += 1
print(liftEdid1)
print(liftEdid2)
print(rightEdid1)
print(rightEdid2)
# %%

# %%
ws.cell(1,2).value
# %%
int(ws.cell(1,2).value,16)
# %%
hex(128)
# %%
if ws.cell(522,1).value == None:
    print('123')
# %%
'{:02x}'.format(16)
# %%
if liftEdid1 == liftEdid2:
    print(123)