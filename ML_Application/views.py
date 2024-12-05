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
import matplotlib
import matplotlib.pyplot as plt
import io
import urllib, base64
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
matplotlib.use('Agg') 

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
@csrf_exempt
@login_required(login_url='/login')
def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('fileUpload')  # FilePond sends the file with key 'file'
        if not uploaded_file:
            return JsonResponse({'error': 'No file uploaded.'}, status=400)

        # Save the file to the user's directory
        user_folder = os.path.join(settings.MEDIA_ROOT, 'datasets', request.user.username)
        os.makedirs(user_folder, exist_ok=True)  # Ensure the user's folder exists
        file_path = os.path.join(user_folder, uploaded_file.name)

        with default_storage.open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # Save the dataset instance to the database
        dataset = Dataset.objects.create(
            name=uploaded_file.name,
            file=f"media/datasets/{request.user.username}/{uploaded_file.name}",
            user=request.user
        )

        # Respond with dataset details
        return redirect('upload')

    # For GET request, render the upload template with datasets
    datasets = Dataset.objects.filter(user=request.user)
    return render(request, 'upload.html', {'datasets': datasets})

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
    user_datasets = Dataset.objects.filter(user=request.user) 

    selected_dataset_id = request.GET.get('dataset_id')  # From dropdown selection
    # Read the dataset into a pandas DataFrame
    
    if selected_dataset_id:
        dataset = get_object_or_404(user_datasets, id=selected_dataset_id)
        file_path = dataset.file.path
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)

        # Compute dataset statistics
        row_count = df.shape[0]
        feature_count = df.shape[1]
        missing_values = df.isnull().sum()
        duplicate_rows = df.duplicated().sum()
        data_types = df.dtypes
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
                        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
                    elif fill_method == 'next':
                        df.fillna(method='ffill', inplace=True)
                    elif fill_method == 'most_frequent':
                        imputer = SimpleImputer(strategy='most_frequent')
                        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
                df.to_csv(dataset.file.path, index=False)

            if 'transform_data' in request.POST:
                transform_type = request.POST.get('transform_type')
                numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
                if transform_type == 'normalize':
                    scaler = MinMaxScaler()
                    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                elif transform_type == 'standardize':
                    scaler = StandardScaler()
                    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                if 'feature_selection' in request.POST:
                    selector = SelectKBest(f_classif, k='all')
                    target_column = 'target'  # Update as per your dataset
                    df = pd.DataFrame(selector.fit_transform(df, df[target_column]), columns=selector.get_feature_names_out())
                df.to_csv(dataset.file.path, index=False)

        return render(request, 'preprocess.html', {
            'head': head,
            'row_count': row_count,
            'feature_count': feature_count,
            'missing_values': missing_values,
            'duplicate_rows': duplicate_rows,
            'data_types': data_types,
            'datasets': user_datasets,
            'selected_dataset_id': selected_dataset_id,
        })

    return render(request, 'preprocess.html', {'datasets': user_datasets})
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
        categorical_vars = [col for col in df.columns if df[col].nunique() <= 20 and df[col].dtype == 'object']
        continuous_vars = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]

        return JsonResponse({
            'categorical_vars': categorical_vars,
            'continuous_vars': continuous_vars
        })

    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=400)
    
def generate_chart(request):
    dataset_id = request.GET.get('dataset_id')
    chart_type = request.GET.get('chart_type')  # 'pie', 'histogram', or 'scatter'
    column_x = request.GET.get('column_x')  # For scatter plot, x-axis column
    column_y = request.GET.get('column_y')  # For scatter plot, y-axis column

    try:
        dataset = Dataset.objects.get(id=dataset_id, user=request.user)
        file_path = dataset.file.path
        
        # Load the dataset
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return JsonResponse({'error': 'Unsupported file format'}, status=400)

        # Prepare a chart buffer
        buf = io.BytesIO()

        fig, ax = plt.subplots(figsize=(4, 3)) 

        if chart_type == 'pie':
            # Pie chart for categorical data
            df[column_x].value_counts().plot(kind='pie', ax=plt.gca(), autopct='%1.1f%%')
            plt.title(f'{column_x} Distribution (Pie Chart)')
            plt.ylabel('')
            plt.savefig(buf, format='png')
            plt.close()

        elif chart_type == 'histogram':
            # Histogram for categorical data
            df[column_x].value_counts().plot(kind='bar', ax=plt.gca())
            plt.title(f'{column_x} Distribution (Histogram)')
            plt.ylabel('Frequency')
            plt.savefig(buf, format='png')
            plt.close()

        elif chart_type == 'scatter':
            # Scatter plot for continuous data
            df.plot.scatter(x=column_x, y=column_y, ax=plt.gca())
            plt.title(f'{column_x} vs {column_y} (Scatter Plot)')
            plt.savefig(buf, format='png')
            plt.close()

        buf.seek(0)
        chart_base64 = base64.b64encode(buf.read()).decode('utf-8')  # Base64 encode the image
        buf.close()

        # Debugging: log the base64 response
        print("Generated chart base64: ", chart_base64[:50])  # Print a snippet of the base64 string for debugging

        return JsonResponse({'chart_base64': chart_base64})

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
