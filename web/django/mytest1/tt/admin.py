from django.contrib import admin


# python manage.py createsuperuser
#python manage.py makemigrations
#python manage.py migrate

from tt.models import Test, Contact, Tag

# Register your models here.
# admin.site.register([Test, Contact, Tag])

# # Register your models here.
# class ContactAdmin(admin.ModelAdmin):
#     fields = ('name', 'email')
#
#
# admin.site.register(Contact, ContactAdmin)
# admin.site.register([Test, Tag])

# Register your models here.
class ContactAdmin(admin.ModelAdmin):
    fieldsets = (
        ['Main', {
            'fields': ('name', 'email'),
        }],
        ['Advance', {
            'classes': ('collapse',),  # CSS
            'fields': ('age',),
        }]
    )


admin.site.register(Contact, ContactAdmin)
admin.site.register([Test, Tag])

