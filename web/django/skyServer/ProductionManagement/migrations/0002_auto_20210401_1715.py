# Generated by Django 3.1.7 on 2021-04-01 09:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ProductionManagement', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='pcb',
            name='hex',
            field=models.FileField(blank=True, upload_to='', verbose_name='程序'),
        ),
    ]
