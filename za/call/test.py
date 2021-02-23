#%%


class A(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __call__(self):
        print('my name is %s' % self.name)
        print('my age is %s' % self.age)


if __name__ == '__main__':
    a = A('jack', 26)
    a()
# %%
x=0.0
t=1.0
for i in range(100):
    x=t*0.3+x
    t=t*0.7*0.5
print(x)
# %%
x
# %%
