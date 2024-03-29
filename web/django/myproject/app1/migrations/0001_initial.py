# Generated by Django 3.1.7 on 2021-03-15 02:57

import app1.models
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Book',
            fields=[
                ('ISBN', models.CharField(max_length=13, primary_key=True, serialize=False, verbose_name='ISBN')),
                ('title', models.CharField(max_length=128, verbose_name='书名')),
                ('author', models.CharField(max_length=32, verbose_name='作者')),
                ('press', models.CharField(max_length=64, verbose_name='出版社')),
                ('description', models.CharField(default='', max_length=1024, verbose_name='详细')),
                ('price', models.CharField(max_length=20, null=True, verbose_name='价格')),
                ('category', models.CharField(default='文学', max_length=64, verbose_name='分类')),
                ('cover', models.ImageField(blank=True, upload_to=app1.models.custom_path, verbose_name='封面')),
                ('index', models.CharField(max_length=16, null=True, verbose_name='索引')),
                ('location', models.CharField(default='图书馆1楼', max_length=64, verbose_name='位置')),
                ('quantity', models.IntegerField(default=1, verbose_name='数量')),
            ],
            options={
                'verbose_name': '图书',
                'verbose_name_plural': '图书',
            },
        ),
    ]
