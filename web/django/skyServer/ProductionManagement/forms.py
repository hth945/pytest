#!/usr/bin/env python
# -*- coding: utf-8 -*-

from django import forms
from .models import PCB


class SearchForm(forms.Form):
    CHOICES = [
        (u'name', u'名称'),
        (u'id', u'id'),
    ]

    search_by = forms.ChoiceField(
        label='',
        choices=CHOICES,
        widget=forms.RadioSelect(),
        initial=u'name',
    )

    keyword = forms.CharField(
        label='',
        max_length=32,
        widget=forms.TextInput(attrs={
            'class': 'form-control input-lg',
            'placeholder': u'请输入需要搜索的id或名称',
            'name': 'keyword',
        })
    )

class upPCBForm(forms.ModelForm):
    class Meta:
        model = PCB
        fields = '__all__'
