from django.shortcuts import render
from django.shortcuts import render, HttpResponse, redirect
from .models import *
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
import os
from django.conf import settings
from django.shortcuts import render, redirect
from .forms import FileUploadForm
from .models import Dataset
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages


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
@login_required(login_url='/login')
def upload(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.cleaned_data['file']

            # Save the dataset instance
            dataset = Dataset.objects.create(
                name=file.name,
                file=file,  # Assign the file directly to the FileField
                user=request.user  # Associate the dataset with the logged-in user
            )

            # Redirect to a success page or the same upload page
            return redirect('upload')

    else:
        form = FileUploadForm()

    return render(request, 'upload.html', {'form': form})
    
#end upload 


# home 
def home(request):
    
    print('bnjrjr')
    return render(request, 'home.html')
#end  home 
 

import matplotlib.pyplot as plt
import io
import urllib, base64
# visualisation 
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
#end visualisation 


# modeles 
def modeles(request):
    
    print('bnjrjr')
    return render(request, 'modeles.html')
#end modeles 


# Predictions 
def Predictions(request):
    
    print('bnjrjr')
    return render(request, 'Predictions.html')
#end Predictions 


# documentation 
def documentation(request):
    
    print('bnjrjr')
    return render(request, 'documentation.html')
#end documentation 

# report 
def report(request):
    
    print('bnjrjr')
    return render(request, 'report.html')
#end report 
