
#%%
# import sys
# print(sys.version) 
# print(sys.path)
import helloworld
print(helloworld.helloworld())

print(helloworld.add_one(1))
#%%
print(helloworld.bytesTest('123'))

#%%

import sys
import matrix

mat_a = matrix.zeros(3,3)
print(mat_a)

print(mat_a.data)  # 打印矩阵中的数据
print(mat_a.to_list())  # 获取矩阵中的数据到list中

print(mat_a.width)  # 打印矩阵中的数据
# %%
# Py_BuildValue("")                        None
# Py_BuildValue("i", 123)                  123
# Py_BuildValue("iii", 123, 456, 789)      (123, 456, 789)
# Py_BuildValue("s", "hello")              'hello'
# Py_BuildValue("y", "hello")              b'hello'
# Py_BuildValue("ss", "hello", "world")    ('hello', 'world')
# Py_BuildValue("s#", "hello", 4)          'hell'
# Py_BuildValue("y#", "hello", 4)          b'hell'
# Py_BuildValue("()")                      ()
# Py_BuildValue("(i)", 123)                (123,)
# Py_BuildValue("(ii)", 123, 456)          (123, 456)
# Py_BuildValue("(i,i)", 123, 456)         (123, 456)
# Py_BuildValue("[i,i]", 123, 456)         [123, 456]
# Py_BuildValue("{s:i,s:i}",
#               "abc", 123, "def", 456)    {'abc': 123, 'def': 456}
# Py_BuildValue("((ii)(ii)) (ii)",
#               1, 2, 3, 4, 5, 6)          (((1, 2), (3, 4)), (5, 6))
#%%

# %%
