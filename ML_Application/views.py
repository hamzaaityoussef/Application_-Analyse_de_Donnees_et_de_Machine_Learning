from django.shortcuts import render
from django.shortcuts import render, HttpResponse, redirect
from .models import *

# Create your views here.
def base(request):
    info = {
        'user_session': 'TestSession',
        'ip_address': '127.0.0.1',
        'current_time': '2024-08-21T14:00:00'
    }
    print('bnjrjr')

    # Call the synchronous function directly
    # send_telegram_message('Run', info)

    return render(request, 'base.html', {'user_role': 'user_role'})

from django.contrib.auth import authenticate, login, logout
from django.contrib import messages



def loginpage(request):
    if request.user.is_authenticated:
        
        return redirect('upload')
        

    if request.method == 'POST':
        email = request.POST.get('username')
        password = request.POST.get('password')

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            user = None

        if user is not None and user.password == password:
            
            login(request, user)
            return redirect('upload')    
        else:
            messages.error(request, 'Nom d\'utilisateur ou mot de passe invalide. Veuillez réessayer.')

    return render(request, 'login.html')





def logoutUser(request):
    logout(request)
    return redirect('login')
