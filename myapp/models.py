from django.db import models


class Document(models.Model):
    objects = None
    docfile = models.FileField(upload_to='documents/%Y/%m/%d')