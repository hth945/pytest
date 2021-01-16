import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
  
filename = 'sen' # 源文件名 生成的pyd文件名  
full_filename = 'test.pyx' # 包含后缀的源文件名
  
setup(
  name = 'test',        #项目名称 
  cmdclass = {'build_ext': build_ext},
  ext_modules=[Extension(filename,sources=[full_filename, "cython_test.c"],
         include_dirs=[numpy.get_include()])],
)