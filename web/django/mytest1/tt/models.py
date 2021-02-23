from django.db import models

# Create your models here.

#python manage.py makemigrations
#python manage.py migrate
class Test(models.Model):
    # title = models.CharField(max_length= 50)
    # author = models.CharField(max_length= 20)
    # time = models.IntegerField(default = 0)
    name = models.CharField(max_length=20)

