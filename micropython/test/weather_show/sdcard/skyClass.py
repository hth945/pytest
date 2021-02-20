import struct

class people:
    #定义基本属性
    name = ''
    age = 0
    #定义私有属性,私有属性在类外部无法直接进行访问
    __weight = 0
    #定义构造方法
    def __init__(self,n,a,w):
        self.name = n
        self.age = a
        self.__weight = w
    def speak(self):
        print("%s 说: 我 %d 岁。" %(self.name,self.age))

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, age):
        self._age = age

 
#单继承示例
class student(people):
    grade = ''
    def __init__(self,n,a,w,g):
        #调用父类的构函
        people.__init__(self,n,a,w)
        self.grade = g
    #覆写父类的方法
    def speak(self):
        print("%s 说: 我 %d 岁了，我在读 %d 年级"%(self.name,self.age,self.grade))
 



class test:
    def __init__(self,):
        a=bytes('hello ',encoding="ascii")
        b=bytes('world!',encoding="ascii")
        c=2
        d=45.123
        self.structBytes=bytearray(struct.pack('4s5sbf',a,b,c,d))

    @property
    def a(self):
        return struct.unpack("4s", self.structBytes[0:4])[0]

    @a.setter
    def a(self, a):
        tem = bytes(a,encoding="ascii")
        self.structBytes[0:4]=bytearray(struct.pack('4s',tem))

# t = test()
# print(t.a)
# t.a='456789'
# print(t.a)

import skyFun


class pyBinTest:
    def __init__(self,):
        self.structBytes=skyFun.getBinData('pyBinTest')

    @property
    def a(self):
        return struct.unpack("i", self.structBytes[0:4])[0]

    @a.setter
    def a(self, a):
        self.structBytes[0:4]=bytearray(struct.pack('i',a))

    @property
    def b(self):
        return struct.unpack("b", self.structBytes[4:5])[0]

    @b.setter
    def b(self, b):
        self.structBytes[4:5]=bytearray(struct.pack('b',b))

    @property
    def c(self):
        return struct.unpack("2s", self.structBytes[5:7])[0]

    @c.setter
    def c(self, c):
        self.structBytes[5:7]=bytearray(struct.pack('2s',c))

    @property
    def d(self):
        return struct.unpack("f", self.structBytes[8:12])[0]

    @d.setter
    def d(self, d):
        self.structBytes[8:12]=bytearray(struct.pack('f',d))


    @property
    def e(self):
        return struct.unpack("5s", self.structBytes[12:17])[0]

    @e.setter
    def e(self, e):
        self.structBytes[12:17]=bytearray(struct.pack('5s',e))


