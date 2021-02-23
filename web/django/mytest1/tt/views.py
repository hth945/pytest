from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse


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
