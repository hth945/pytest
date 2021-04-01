from django.contrib import admin

# Register your models here.
from .models import PCB

admin.site.register([PCB])