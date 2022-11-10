from django.db import models


class Document(models.Model):
    objects = None
    doc_file = models.FileField(upload_to='documents/%Y/%m/%d')
