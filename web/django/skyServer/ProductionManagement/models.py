from django.db import models

# Create your models here.
from django.contrib.auth.models import User
import uuid, os

def custom_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format(uuid.uuid4().hex[:10], ext)
    return filename


class PCB(models.Model):
    name = models.CharField(max_length=64,unique=True,verbose_name='名称')
    directions = models.TextField(verbose_name='说明')
    config = models.TextField(verbose_name='配置')
    date_issued = models.DateField(verbose_name='上传时间')
    # hex = models.BinaryField(verbose_name='程序')
    hex = models.FileField(verbose_name='程序',upload_to=custom_path,blank=True)

