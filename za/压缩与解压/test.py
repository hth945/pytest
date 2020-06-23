#%%
from glob import glob
import zipfile
import os

#定义一个函数，递归读取absDir文件夹中所有文件，并塞进zipFile文件中。参数absDir表示文件夹的绝对路径。
def writeAllFileToZip(absDir,zipFile):
    for f in os.listdir(absDir):
        absFile=os.path.join(absDir,f) #子文件的绝对路径
        if os.path.isdir(absFile): #判断是文件夹，继续深度读取。
            relFile=absFile[len(os.getcwd())+1:] #改成相对路径，否则解压zip是/User/xxx开头的文件。
            zipFile.write(relFile) #在zip文件中创建文件夹
            writeAllFileToZip(absFile,zipFile) #递归操作
        else: #判断是普通文件，直接写到zip文件中。
            relFile=absFile[len(os.getcwd())+1:] #改成相对路径
            zipFile.write(relFile)
    return

zipFile=zipfile.ZipFile('./1.zip',"w",zipfile.ZIP_DEFLATED) 
files = glob(r'./1/*.txt')
for file in files:
    #写入要压缩文件，并添加归档文件名称
    print(file)
    zipFile.write(file)
zipFile.close()
# %%


# %%

# %%
