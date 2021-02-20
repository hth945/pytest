import os

def rmdir(directory):
    os.chdir(directory)
    for f in os.listdir():
        try:
            os.remove(f)
        except OSError:
            pass
    for f in os.listdir():
        rmdir(f)
    os.chdir('..')
    os.rmdir(directory)
rmdir("/sdcard") # 删除一个文件


os.chdir('./emmc') # 修改当前工作目录
print(os.getcwd()) # 查看当前工作目录
for f in os.listdir():
    print(f)


with open("123.txt","w+") as fil:
    fil.write("123")


with open("123.txt","r") as fil:
    data = fil.read()
    print(data)

	
os.remove("123.txt") # 删除一个文件

