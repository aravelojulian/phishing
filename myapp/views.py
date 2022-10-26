from django.shortcuts import redirect, render
from .models import Document
from .forms import DocumentForm

import logging
logger = logging.getLogger(__name__)


def index(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()

            return redirect('index')
    else:
        form = DocumentForm()

    document = Document.objects.last()

    context = {'document': document, 'form': form}
    return render(request, 'index.html', context)
