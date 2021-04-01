from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from .forms import SearchForm,upPCBForm
from .models import PCB
from django.contrib import auth
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger


def hello(request):
    return HttpResponse("Hello world ! ")

def index(request):
    context = {
        'searchForm': SearchForm(),
    }

    return render(request, 'index.html', context)
    # context          = {}
    # context['hello'] = 'Hello World!'
    # return render(request, 'index.html', context)

def PCB_search(request):
    search_by = request.GET.get('search_by', 'name')
    PCBs = []
    current_path = request.get_full_path()

    keyword = request.GET.get('keyword', u'_书目列表')

    if keyword == u'_板子列表':
        PCBs = PCB.objects.all()
    else:
        if search_by == u'name':
            keyword = request.GET.get('keyword', None)
            PCBs = PCB.objects.filter(name__contains=keyword).order_by('-id')[0:50]
        elif search_by == u'id':
            keyword = request.GET.get('keyword', None)
            PCBs = PCB.objects.filter(id__contains=keyword).order_by('-id')[0:50]

    paginator = Paginator(PCBs, 5)
    page = request.GET.get('page', 1)

    try:
        PCBs = paginator.page(page)
    except PageNotAnInteger:
        PCBs = paginator.page(1)
    except EmptyPage:
        PCBs = paginator.page(paginator.num_pages)

    # ugly solution for &page=2&page=3&page=4
    if '&page' in current_path:
        current_path = current_path.split('&page')[0]

    context = {
        'PCBs': PCBs,
        'search_by': search_by,
        'keyword': keyword,
        'current_path': current_path,
        'searchForm': SearchForm(),
    }
    return render(request, 'search.html', context)


def upPCB(request):
    context = {
        'upPCBForm': upPCBForm(),
    }

    tem = upPCBForm()
    print(tem)
    return render(request, 'upPCB.html', context)