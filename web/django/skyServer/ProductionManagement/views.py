from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from .forms import SearchForm,upPCBForm, LoginForm, RegisterForm, ResetPasswordForm
from .models import PCB
from django.contrib import auth
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseForbidden
from django.shortcuts import redirect

def hello(request):
    return HttpResponse("Hello world ! ")

def index(request):
    # auth.logout(request)
    context = {
        'searchForm': SearchForm(),
    }

    return render(request, 'index.html', context)



def user_login(request):
    if request.user.is_authenticated:
        return HttpResponseRedirect('/')


    state = None

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = auth.authenticate(username=username, password=password)

        if user:
            if user.is_active:
                auth.login(request, user)
                return HttpResponseRedirect('/')
            else:
                return HttpResponse(u'Your account is disabled.')
        else:
            state = 'not_exist_or_password_error'

    context = {
        'loginForm': LoginForm(),
        'state': state,
    }

    return render(request, 'login.html', context)

def user_logout(request):
    auth.logout(request)
    return HttpResponseRedirect('/')

def user_register(request):
    if request.user.is_authenticated:
        return HttpResponseRedirect('/')

    registerForm = RegisterForm()

    state = None
    if request.method == 'POST':
        registerForm = RegisterForm(request.POST, request.FILES)
        password = request.POST.get('password', '')
        repeat_password = request.POST.get('re_password', '')
        if password == '' or repeat_password == '':
            state = 'empty'
        elif password != repeat_password:
            state = 'repeat_error'
        else:
            username = request.POST.get('username', '')

            if User.objects.filter(username=username):
                state = 'user_exist'
            else:
                user_obj = User.objects.create_user(username=username,password=password)
                state = 'success'

                auth.login(request, user_obj)

                context = {
                    'state': state,
                    'registerForm': registerForm,
                }
                return render(request, 'register.html', context)

    context = {
        'state': state,
        'registerForm': registerForm,
    }

    return render(request, 'register.html', context)

@login_required
def set_password(request):
    user = request.user
    state = None
    if request.method == 'POST':
        old_password = request.POST.get('old_password', '')
        new_password = request.POST.get('new_password', '')
        repeat_password = request.POST.get('repeat_password', '')

        if user.check_password(old_password):
            if not new_password:
                state = 'empty'
            elif new_password != repeat_password:
                state = 'repeat_error'
            else:
                user.set_password(new_password)
                user.save()
                state = 'success'

    context = {
        'state': state,
        'resetPasswordForm': ResetPasswordForm(),
    }

    return render(request, 'set_password.html', context)


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

    paginator = Paginator(PCBs, 15)
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

@login_required(login_url='/login/')
def upPCB(request):
    data = {}
    if request.method == "POST":
        print('123')
        pcb = upPCBForm(request.POST)
        data['upPCBForm'] = pcb
        if pcb.is_valid():
            new_book = pcb.save()
            data['upresult']= '保存成功'
        else:
            data['upresult'] = '保存失败'
    else:
        data['upPCBForm'] = upPCBForm()
        data['upresult'] = ''

    return render(request, 'upPCB.html', data)


def about(request):
    return render(request, 'about.html', {})


#构造钉钉登录url
def ding_url(request):
    appid = 'dingoa6jgust3skqrq9hak'
    redirect_uri = 'http://hth945.xyz:8000/'
    return redirect('https://oapi.dingtalk.com/connect/qrconnect?appid='+appid+'&response_type=code&scope=snsapi_login&state=STATE&redirect_uri='+redirect_uri)