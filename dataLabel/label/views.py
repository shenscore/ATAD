from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.decorators import api_view


def index(request):
    return render(request,'index.html')

