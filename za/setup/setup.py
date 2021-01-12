from distutils.core import setup, Extension
from shutil import copyfile

name='helloworld'
bname = '.cp37-win_amd64.pyd'
setup(name=name, version='1.1', ext_modules=[Extension(name, ['hello.c'])])
copyfile('build/lib.win-amd64-3.7/'+name+bname, name+bname)

import helloworld
print(helloworld.helloworld())
print(helloworld.add_one(1))
b=bytes('123',encoding="ASCII")
print(helloworld.bytesTest(b))
print(b)

# setup(name='matrix', version='1.0', ext_modules=[Extension('matrix', ['matrix.c'])])