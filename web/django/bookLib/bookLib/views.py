from django.http import HttpResponse
from bookLib.forms import SearchForm
from django.shortcuts import render
from bookLib.models import Book

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

def book_search(request):
    search_by = request.GET.get('search_by', '书名')
    books = []
    current_path = request.get_full_path()

    keyword = request.GET.get('keyword', u'_书目列表')

    if keyword == u'_书目列表':
        books = Book.objects.all()
    else:
        if search_by == u'书名':
            keyword = request.GET.get('keyword', None)
            books = Book.objects.filter(title__contains=keyword).order_by('-title')[0:50]
        elif search_by == u'ISBN':
            keyword = request.GET.get('keyword', None)
            books = Book.objects.filter(ISBN__contains=keyword).order_by('-title')[0:50]
        elif search_by == u'作者':
            keyword = request.GET.get('keyword', None)
            books = Book.objects.filter(author__contains=keyword).order_by('-title')[0:50]

    paginator = Paginator(books, 5)
    page = request.GET.get('page', 1)

    try:
        books = paginator.page(page)
    except PageNotAnInteger:
        books = paginator.page(1)
    except EmptyPage:
        books = paginator.page(paginator.num_pages)

    # ugly solution for &page=2&page=3&page=4
    if '&page' in current_path:
        current_path = current_path.split('&page')[0]

    context = {
        'books': books,
        'search_by': search_by,
        'keyword': keyword,
        'current_path': current_path,
        'searchForm': SearchForm(),
    }
    return render(request, 'library/search.html', context)


def user_login(request):
    # if request.user.is_authenticated:
    #     return HttpResponseRedirect('/')

    # state = None

    # if request.method == 'POST':
    #     username = request.POST.get('username')
    #     password = request.POST.get('password')

    #     user = auth.authenticate(username=username, password=password)

    #     if user:
    #         if user.is_active:
    #             auth.login(request, user)
    #             return HttpResponseRedirect('/')
    #         else:
    #             return HttpResponse(u'Your account is disabled.')
    #     else:
    #         state = 'not_exist_or_password_error'

    # context = {
    #     'loginForm': LoginForm(),
    #     'state': state,
    # }
    context=[]
    return render(request, 'library/login.html', context)
