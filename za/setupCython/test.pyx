#添加cython 才能使语法高亮

#.pyx中用cdef定义的东西，除类以外对.py都是不可见的；
#.py中是不能操作C类型的，如果想在.py中操作C类型就要
# 在.pyx中从python object转成C类型或者用含有set/get方法的C类型包裹类；

from libc.stdlib cimport malloc, free
cdef extern from "cython_test.h":
    struct person_info_t:
        int age
        unsigned char *gender

    ctypedef person_info_t person_info
    cpdef int add_number(int a, int b)
    void print_person_info(char *name, person_info *info)


def say_hello():
    print("hello world")



def cyprint_person_info(name, info):
    cdef person_info pinfo
    pinfo.age = info.age
    pinfo.gender = info.gender
    print_person_info(name, &pinfo)


# 编写包装结构体的类

cdef class person_info_wrap(object):
    cdef person_info *ptr
    
    def __cinit__(self):
        self.ptr = <person_info *>malloc(sizeof(person_info))
    
    def __del__(self):
        free(self.ptr)

    @property
    def age(self):
        return self.ptr.age
    @age.setter
    def age(self, value):
        self.ptr.age = value
    
    @property   #装饰器
    def gender(self):
        return self.ptr.gender

    @gender.setter
    def gender(self, value):
        self.ptr.gender = value

            