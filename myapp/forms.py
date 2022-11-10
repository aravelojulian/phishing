from django import forms


class DocumentForm(forms.Form):
    doc_file = forms.FileField(label='Select a file')
