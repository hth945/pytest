

# 引用cython 编写API扩展库

#编译命令 python setup.py build_ext --inplace

from sen import add_number,cyprint_person_info,say_hello,person_info_wrap


if __name__ == '__main__':
    info_wrap = person_info_wrap()
    say_hello()
    info_wrap.age = 88
    print(info_wrap.age)
    # info_wrap.gender = 'mmmale'
    # cyprint_person_info('hhhandsome', info_wrap)
    




# %%
