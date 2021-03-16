
from django.db import models
from django.contrib.auth.models import User
import uuid, os

def custom_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format(uuid.uuid4().hex[:10], ext)
    return filename

class Book(models.Model):
    class Meta:
        verbose_name = '图书'
        verbose_name_plural = '图书'

    ISBN = models.CharField(max_length=13, primary_key=True, verbose_name='ISBN')
    title = models.CharField(max_length=128, verbose_name='书名')
    author = models.CharField(max_length=32, verbose_name='作者')
    press = models.CharField(max_length=64, verbose_name='出版社')

    description = models.CharField(max_length=1024, default='', verbose_name='详细')
    price = models.CharField(max_length=20, null=True, verbose_name='价格')

    category = models.CharField(max_length=64, default=u'文学', verbose_name='分类')
    cover = models.ImageField(blank=True, upload_to=custom_path, verbose_name='封面')
    index = models.CharField(max_length=16, null=True, verbose_name='索引')
    location = models.CharField(max_length=64, default=u'图书馆1楼', verbose_name='位置')
    quantity = models.IntegerField(default=1, verbose_name='数量')

    def __str__(self):
        return self.title + self.author