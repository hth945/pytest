from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from tt import models
from tt.My_forms import EmpForm


def hello(request):
    return HttpResponse("Hello world ! ")


def runoob(request):
    context          = {"name1": 0,"name2":"菜鸟教程","views_str": "<a href='https://www.runoob.com/'>点击跳转</a>"}
    context['hello'] = 'Hello World123123!'
    context['name'] = 'hth'
    context['views_list']  = ["菜鸟教程1", "菜鸟教程2", "菜鸟教程3"]
    context['views_dict']  = {"name": "菜鸟教程"}
    return render(request, 'runoob.html', context)

def inherit(request):
    return render(request, 'inherit.html')

# def add_book(request):
#     book = models.Book(title="菜鸟教程",price=300,publish="菜鸟出版社",pub_date="2008-8-8")
#     book.save()
#     return HttpResponse("<p>数据添加成功！</p>")

# def add_book(request):
#     books = models.Book.objects.create(title="如来神掌",price=200,publish="功夫出版社",pub_date="2010-10-10")
#     print(books, type(books)) # Book object (18)
#     return HttpResponse("<p>数据添加成功！</p>")


def add_book(request):
    books = models.Book.objects.all()
    print(books,type(books)) # QuerySet类型，类似于list，访问 url 时数据显示在命令行窗口中。
    return HttpResponse("<p>查找成功！</p>")

def add_emp(request):
    if request.method == "GET":
        form = EmpForm()
        return render(request, "add_emp.html", {"form": form})
    else:
        form = EmpForm(request.POST)
        if form.is_valid():  # 进行数据校验
            # 校验成功
            data = form.cleaned_data  # 校验成功的值，会放在cleaned_data里。
            # data.pop('r_salary')
            data.pop('salary')
            print(data)

            models.Emp.objects.create(**data)
            return HttpResponse(
                'ok'
            )
            # return render(request, "add_emp.html", {"form": form})
        else:
            print(form.errors)    # 打印错误信息
            clean_errors = form.errors.get("__all__")
            print(222, clean_errors)
        return render(request, "add_emp.html", {"form": form, "clean_errors": clean_errors})