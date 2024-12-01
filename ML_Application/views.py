from django.shortcuts import render, HttpResponse, redirect, get_object_or_404
from .models import *
import pandas as pd
from django.http import JsonResponse
import os
from django.conf import settings
from .forms import FileUploadForm
from .models import Dataset
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
import matplotlib.pyplot as plt
import io
import urllib, base64

@login_required(login_url='/login')
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




# login and logout
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

# end login and logout

# home 
def home(request):
    
    print('bnjrjr')
    return render(request, 'home.html')
#end  home 



# upload and delete datasets
@login_required(login_url='/login')
def upload(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.cleaned_data['file']

            # Save the dataset instance
            dataset = Dataset.objects.create(
                name=file.name,
                file=file,
                user=request.user
            )

            # Redirect to refresh the page
            return redirect('upload')

    else:
        form = FileUploadForm()

    # Fetch datasets associated with the logged-in user
    datasets = Dataset.objects.filter(user=request.user)

    return render(request, 'upload.html', {'form': form, 'datasets': datasets})


@login_required
def delete_dataset(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id, user=request.user)

    # Delete the file from storage
    if dataset.file:
        file_path = dataset.file.path
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Delete the database entry
    dataset.delete()

    # Redirect back to the upload page
    return redirect('upload')

#end upload 


# Preprocess 
from django.shortcuts import render, redirect
from .models import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

def Preprocess(request):
    dataset = Dataset.objects.get(id=9)  # Replace with actual dataset logic
    df = pd.read_csv(dataset.file.path)  # Load dataset

    # Basic Statistics
    row_count = df.shape[0]
    feature_count = df.shape[1]
    missing_values = df.isnull().sum()
    duplicate_rows = df.duplicated().sum()
    data_types = df.dtypes

    # First 5 rows
    head = df.head(10).to_html(classes="table table-light")

    if request.method == 'POST':
        if 'clean_data' in request.POST:
            action = request.POST.get('action')
            if action == 'delete':
                df = df.dropna()

            elif action == 'fill':
                fill_method = request.POST.get('fill_method')
                if fill_method == 'mean':
                    imputer = SimpleImputer(strategy='mean')
                elif fill_method == 'next':
                    df.fillna(method='ffill', inplace=True)
                elif fill_method == 'most_frequent':
                    imputer = SimpleImputer(strategy='most_frequent')
                df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

            df.to_csv(dataset.file.path)  # Save updated dataset

        if 'transform_data' in request.POST:
            transform_type = request.POST.get('transform_type')
            if transform_type == 'normalize':
                scaler = MinMaxScaler()
                df[df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(
                    df.select_dtypes(include=['float64', 'int64']))
            elif transform_type == 'standardize':
                scaler = StandardScaler()
                df[df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(
                    df.select_dtypes(include=['float64', 'int64']))
            
            # Feature selection logic
            if 'feature_selection' in request.POST:
                selector = SelectKBest(f_classif, k='all')
                df = selector.fit_transform(df, df['target'])  # Assuming target column is 'target'

            df.to_excel(dataset.file.path)  # Save transformed dataset

    return render(request, 'preprocess.html', {
        'head': head,
        'row_count': row_count,
        'feature_count': feature_count,
        'missing_values': missing_values,
        'duplicate_rows': duplicate_rows,
        'data_types': data_types,
        'dataset': dataset,
    })

#end  Preprocess 





from django.shortcuts import render
from django.http import JsonResponse
from .models import Dataset
import pandas as pd
import os
# visualisation 
@login_required(login_url='/login')

# View to render the visualisation page
def visualisation(request):
    datasets = Dataset.objects.filter(user=request.user)  # Get datasets for the logged-in user
    return render(request, 'visualisation.html', {'datasets': datasets})

# View to handle AJAX request for dataset columns
def get_columns(request):
    dataset_id = request.GET.get('dataset_id')  # Get the dataset ID from the request
    try:
        dataset = Dataset.objects.get(id=dataset_id, user=request.user)  # Ensure it's the user's dataset
        file_path = dataset.file.path  # Get the file path for the selected dataset
        
        # Read the dataset into a pandas DataFrame
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return JsonResponse({'error': 'Unsupported file format'}, status=400)
        
        # Get columns from the DataFrame
        columns = df.columns.tolist()
        return JsonResponse({'columns': columns})

    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=400)





#end visualisation 


# modeles 
@login_required(login_url='/login')
def modeles(request):
    
    print('bnjrjr')
    return render(request, 'modeles.html')
#end modeles 


# Predictions 
@login_required(login_url='/login')
def Predictions(request):
    
    print('bnjrjr')
    return render(request, 'Predictions.html')
#end Predictions 


# documentation 
@login_required(login_url='/login')
def documentation(request):
    
    print('bnjrjr')
    return render(request, 'documentation.html')
#end documentation 

# report 
@login_required(login_url='/login')
def report(request):
    
    print('bnjrjr')
    return render(request, 'report.html')
#end report 
