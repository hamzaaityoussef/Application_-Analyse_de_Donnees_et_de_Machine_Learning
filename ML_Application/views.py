from django.shortcuts import render
from django.shortcuts import render, HttpResponse, redirect
from .models import *

def base(request):
    # Check if the user is authenticated
    if request.user.is_authenticated:
        nom = request.user.last_name if request.user.last_name else "Not provided"
        prenom = request.user.first_name if request.user.first_name else "Not provided"
    else:
        nom = "Guest"
        prenom = "User"

    return render(request, 'base.html', {
        'nom': nom,
        'prenom': prenom,
        'user': request.user,  # Pass the full user object for additional details
    })

from django.contrib.auth import authenticate, login, logout
from django.contrib import messages


# login 
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

# end login 


# upload 
def upload(request):
    
    print('bnjrjr')
    return render(request, 'upload.html')
#end upload 


# upload 
def home(request):
    
    print('bnjrjr')
    return render(request, 'home.html')
#end upload 


# upload 
def visualisation(request):
    
    print('bnjrjr')
    return render(request, 'visualisation.html')
#end upload 


# upload 
def modeles(request):
    
    print('bnjrjr')
    return render(request, 'modeles.html')
#end upload 


# upload 
def Predictions(request):
    
    print('bnjrjr')
    return render(request, 'Predictions.html')
#end upload 


# upload 
def documentation(request):
    
    print('bnjrjr')
    return render(request, 'documentation.html')
#end upload 
