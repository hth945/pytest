import array

a = array.array('i', [2, 4, 1, 5])
b = array.array('f')
print(a)
print(b)

a = array.array('f', [3, 6])
print(a)
a.append(7.0)
print(a)

a = array.array('i', [1, 2, 3])
b = array.array('i', [4, 5])
a.extend(b)
print(a)