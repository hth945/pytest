from django import forms


class SearchForm(forms.Form):
        CHOICES = [
            (u'ISBN', u'ISBN'),
            (u'书名', u'书名'),
            (u'作者', u'作者')
        ]

        search_by = forms.ChoiceField(
            label='',
            choices=CHOICES,
            widget=forms.RadioSelect(),
            initial=u'书名',
        )

        keyword = forms.CharField(
            label='',
            max_length=32,
            widget=forms.TextInput(attrs={
                'class': 'form-control input-lg',
                'placeholder': u'请输入需要检索的图书信息',
                'name': 'keyword',
            })
        )