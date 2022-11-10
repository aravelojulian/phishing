from django.shortcuts import render
from .forms import DocumentForm
from .static.scripts.ProcessDataset import ProcessDataset


def index(request):
    # if request.method == 'POST':
    #     form = DocumentForm(request.POST, request.FILES)
    #     print(request.FILES)
    #     # data = ProcessDataset.load_file(request.FILES['doc_file'])
    #     # print(data)
    # else:
    #     form = DocumentForm()

    # context = {'form': form}
    context = {}
    return render(request, 'index.html', context)
