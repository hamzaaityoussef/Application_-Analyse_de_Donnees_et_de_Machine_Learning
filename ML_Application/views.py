from django.shortcuts import render
from django.shortcuts import render, HttpResponse, redirect
from .models import *
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse

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

import matplotlib.pyplot as plt
import io
import urllib, base64
# upload 
def visualisation(request):
    
        csv_path = r'C:\Users\Asus PC\Downloads\Startups.csv'  # Chemin absolu ou relatif du fichier CSV
        df = pd.read_csv(csv_path)  # Charger le CSV dans un DataFrame


        sections = {}
        for i in range(1, 5):
            selected_x = request.GET.get(f'x_axis_{i}', df.columns[0])  # Default to first column
            selected_y = request.GET.get(f'y_axis_{i}', df.columns[1])  # Default to second column

            # Generate the plot for this section
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.scatter(df[selected_x], df[selected_y], color='blue', alpha=0.7)  # Scatter plot
            ax.set_xlabel(selected_x)
            ax.set_ylabel(selected_y)
            ax.set_title(f'Section {i}: {selected_y} vs {selected_x}')

            # Save the plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri = 'data:image/png;base64,' + urllib.parse.quote(string)
            buf.close()

            # Store dropdown values and graph for this section
            sections[i] = {
                'columns': df.columns,
                'selected_x': selected_x,
                'selected_y': selected_y,
                'chart_uri': uri,
            }

        return render(request, 'visualisation.html', {'sections': sections})
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

# upload 
def report(request):
    
    print('bnjrjr')
    return render(request, 'report.html')
#end upload 
