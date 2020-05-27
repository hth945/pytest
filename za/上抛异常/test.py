try:
    print('正常执行')
    # 根据业务逻辑判断，需要手动抛出异常
    #raise Exception(print(a))
    raise Exception('print(a)')#注意这两个的区别，这个带字符串，直接打印字符串里的内容，python把字符串的内容一字不差解析成了异常并打印出来
    print('正常结束')
except Exception as e:
    print('出现异常:', e)

print('OVER')