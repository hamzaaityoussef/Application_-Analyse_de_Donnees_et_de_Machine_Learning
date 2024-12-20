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
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder
from django.shortcuts import render, redirect
from .models import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

from django.http import JsonResponse

from django.shortcuts import render, get_object_or_404
import pandas as pd
from django.http import JsonResponse
import pandas as pd
from sklearn.impute import SimpleImputer

from django.http import JsonResponse, Http404
from django.shortcuts import render
from django.http import JsonResponse
from .models import Dataset
import pandas as pd
import os

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
        
        return redirect('documentation')
        

    if request.method == 'POST':
        email = request.POST.get('username')
        password = request.POST.get('password')

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            user = None

        if user is not None and user.password == password:
            
            login(request, user)
            return redirect('documentation')    
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
        if 'create_dataset' in request.POST:  # Check if the user is creating a dataset
            try:
                # Extract form data for dataset creation
                rows = int(request.POST.get('num_rows'))
                columns = int(request.POST.get('num_columns'))
                namefile = request.POST.get('file_name')
                column_data = []
                for i in range(1, columns + 1):
                    col_name = request.POST.get(f'col{i}_name')
                    col_type = request.POST.get(f'col{i}_type')
                    column_data.append({'name': col_name, 'type': col_type})

                # Create a blank DataFrame
                data = []
                for r in range(rows):
                    row = []
                    for c in range(columns):
                        value = request.POST.get(f'row{r}_col{c}')
                        row.append(value)
                    data.append(row)

                df = pd.DataFrame(data, columns=[col['name'] for col in column_data])

                # Save the DataFrame to the user's directory
                user_folder = os.path.join(settings.MEDIA_ROOT, 'datasets', request.user.username)
                os.makedirs(user_folder, exist_ok=True)  # Ensure the user's folder exists
                file_name = f"{namefile}.csv"
                file_path = os.path.join(user_folder, file_name)
                df.to_csv(file_path, index=False)

                # Save the dataset instance to the database
                dataset = Dataset.objects.create(
                    name=file_name,
                    file=f"datasets/{request.user.username}/{file_name}",
                    user=request.user
                )

                # Save a copy of the dataset
                copy_name = f"{file_name.rsplit('.', 1)[0]}_copy.csv"
                copy_file_path = os.path.join(user_folder, copy_name)
                df.to_csv(copy_file_path, index=False)

                DatasetCopy.objects.create(
                    original_dataset=dataset,
                    name=copy_name,
                    file=f"datasets/{request.user.username}/{copy_name}",
                    user=request.user
                )

                # Log the action
                action = Historique(
                    action='create Data',
                    date_action=datetime.now(),
                    user=request.user,
                    infos=f"{file_name} created successfully"
                )
                action.save()

                messages.success(request, 'Dataset created successfully.')
                return redirect('upload')
            except Exception as e:
                messages.error(request, f"Error creating dataset: {e}")
                return redirect('upload')

        else:  # Handle file upload logic
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
                file=f"datasets/{request.user.username}/{uploaded_file.name}",
                user=request.user
            )

            # Create a copy of the dataset
            base_name, extension = uploaded_file.name.rsplit('.', 1)
            copy_name = f"{base_name}_copy.{extension}"
            copy_file_path = os.path.join(user_folder, copy_name)

            with default_storage.open(copy_file_path, 'wb+') as copy_destination:
                for chunk in uploaded_file.chunks():
                    copy_destination.write(chunk)

            DatasetCopy.objects.create(
                original_dataset=dataset,
                name=copy_name,
                file=f"datasets/{request.user.username}/{copy_name}",
                user=request.user
            )

            # Log the action
            action = Historique(
                action='upload Data',
                date_action=datetime.now(),
                user=request.user,
                infos=f"{uploaded_file.name} uploaded successfully"
            )
            action.save()

            messages.success(request, 'Dataset uploaded successfully.')
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
    action = Historique(
            action='Delete Data',
            date_action=datetime.now(),
            user=request.user,
            infos=f"{dataset.name} deleted successfully" 
            
        )
    action.save()
    messages.success(request, 'data deleted successfully.')

    # Redirect back to the upload page
    return redirect('upload')

#end upload 


# Preprocess 


def preprocess(request):
    user_datasets = DatasetCopy.objects.filter(user=request.user)
    selected_dataset_id = request.GET.get('dataset_id')  # Extract dataset_id from the query parameters

    # Initialize variables
    head = ''
    row_count = 0
    feature_count = 0
    missing_values = {}
    duplicate_rows = 0
    data_types = {}
    columns = []
    status_normalized = False
    status_standardized = False
    status_cleaned = False
    status_encoded = False

    # If a dataset_id is provided, fetch its details
    if selected_dataset_id:
        try:
            dataset = user_datasets.get(id=selected_dataset_id)
            file_path = dataset.file.path

            # Load the dataset
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, engine='openpyxl')

            # Convert datetime fields to strings for compatibility with templates
            for col in df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns:
                df[col] = df[col].astype(str)

            # Compute dataset statistics
            row_count = df.shape[0]
            feature_count = df.shape[1]
            missing_values = df.isnull().sum().to_dict()
            duplicate_rows = df.duplicated().sum()
            data_types = df.dtypes.apply(lambda x: str(x)).to_dict()
            head = df.head(10).to_dict(orient='records')  # First 10 rows as list of dictionaries
            columns = df.columns.tolist()

            # Fetch the statuses from the dataset
            status_normalized = dataset.status_normalized
            status_standardized = dataset.status_standardized
            status_cleaned = dataset.status_cleaned
            status_encoded = dataset.status_encoded

        except DatasetCopy.DoesNotExist:
            # Handle cases where the dataset_id does not match any record
            raise Http404("Dataset not found")
    print('statue cleaned :',status_cleaned)
    return render(request, 'preprocess.html', {
        'head': head,
        'columns': columns,
        'row_count': row_count,
        'feature_count': feature_count,
        'missing_values': missing_values,
        'duplicate_rows': duplicate_rows,
        'data_types': data_types,
        'datasets': user_datasets,
        'selected_dataset_id': selected_dataset_id,
        'status_normalized': status_normalized,
        'status_standardized': status_standardized,
        'status_cleaned': status_cleaned,
        'status_encoded': status_encoded,
    })



@csrf_exempt
def apply_actions(request):
    try:
        # Extract dataset ID from the query parameters
        dataset_id = request.POST.get('dataset_id')
        if not dataset_id:
            return JsonResponse({'error': 'No dataset selected'}, status=400)
        print(request.POST)
        # Fetch the dataset object
        user_datasets = DatasetCopy.objects.filter(user=request.user)
        try:
            dataset = user_datasets.get(id=dataset_id)
        except DatasetCopy.DoesNotExist:
            return JsonResponse({'error': 'Dataset not found'}, status=404)

        # Load the dataset
        file_path = dataset.file.path
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')

        # Handle actions
        updated_data = {}
        if 'clean_data' in request.POST:
            action = request.POST.get('action')
            if action == 'delete':
                df = df.dropna()
                dataset.status_cleaned = True  # Update the status
                print(df.isnull().sum())
            elif action == 'duplicated':
                df = df.drop_duplicates()
                dataset.status_cleaned = True  # Update the status 
                print('somme duplicated :', df.duplicated().sum())
            elif action == 'fill':
                fill_method = request.POST.get('fill_method')
                if fill_method == 'mean':
                    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
                    non_numeric_columns = df.select_dtypes(exclude=['float64', 'int64']).columns
                
                    if not numeric_columns.empty:
                        imputer_numeric = SimpleImputer(strategy='mean')
                        df_numeric = pd.DataFrame(imputer_numeric.fit_transform(df[numeric_columns]), columns=numeric_columns)
                    else:
                        df_numeric = pd.DataFrame()
                    
                    if not non_numeric_columns.empty:
                        imputer_categorical = SimpleImputer(strategy='most_frequent')
                        df_categorical = pd.DataFrame(imputer_categorical.fit_transform(df[non_numeric_columns]), columns=non_numeric_columns)
                    else:
                        df_categorical = pd.DataFrame()
                
                    df = pd.concat([df_numeric, df_categorical], axis=1)
                elif fill_method == 'most_frequent':
                    df.fillna(df.mode().iloc[0], inplace=True)
                elif fill_method == 'next':
                    df.fillna(method='ffill', inplace=True)
                dataset.status_cleaned = True  # Update the status

            elif action == 'drop_column':
                column_to_drop = request.POST.get('column')
                if column_to_drop in df.columns:
                    df = df.drop(columns=[column_to_drop])
                else:
                    return JsonResponse({'error': 'Column not found'}, status=400)

            action = Historique(
                action='Clean Data',
                date_action=datetime.now(),
                user=request.user,
                infos=f"{dataset.name} Cleaned successfully" 
                
                )
            action.save()
            message = 'Data cleaned successfully.'
        
        if 'transform_data' in request.POST:
            transform_type = request.POST.get('transform_type')
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            if transform_type == 'normalize':
                scaler = MinMaxScaler()
                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                dataset.status_normalized = True  # Update the status
                dataset.status_standardized = False
            elif transform_type == 'standardize':
                scaler = StandardScaler()
                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                dataset.status_standardized = True  # Update the status
                dataset.status_normalized = False 

            action = Historique(
                action='Transform Data',
                date_action=datetime.now(),
                user=request.user,
                infos=f"{dataset.name} Transformed successfully" 
                
                )
            action.save()  
            message = 'Data transformed successfully.' 

        # Save the updated dataset back to file
        if file_path.endswith('.csv'):
            df.to_csv(file_path, index=False)
        elif file_path.endswith('.xlsx'):
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
        
        # Save the updated dataset object
        dataset.save()

        # Compute updated statistics for the relevant parts
        updated_data['row_count'] = int(df.shape[0])
        updated_data['feature_count'] = int(df.shape[1])
        updated_data['columns'] = df.columns.tolist()
        updated_data['data_types'] = {key: str(value) for key, value in df.dtypes.to_dict().items()}
        updated_data['head'] = df.head(10).to_dict(orient='records')
        updated_data['missing_values'] = {key: int(value) for key, value in df.isnull().sum().to_dict().items()}
        updated_data['duplicate_rows'] = int(df.duplicated().sum())
        updated_data['status'] = {
            'normalized': dataset.status_normalized,
            'standardized': dataset.status_standardized,
            'cleaned': dataset.status_cleaned,
            
        }
        updated_data['message'] = message


        # Return a JSON response with updated data
        return JsonResponse(updated_data, status=200)
    except Exception as e:
        # Return a JSON response with the error message
        return JsonResponse({"error": str(e)}, status=400) 


#end  Preprocess 






# visualisation 

# View to render the visualisation page
@login_required(login_url='/login')
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
        categorical_vars = [col for col in df.columns if df[col].nunique() <= 20]
        continuous_vars = [col for col in df.columns if col not in categorical_vars and df[col].dtype in ['float64', 'int64']]


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

        # Directory structure for charts
        user_folder = os.path.join(settings.MEDIA_ROOT, 'charts', request.user.username)
        os.makedirs(user_folder, exist_ok=True)

        # Filepath for the chart
        chart_file_name = f"{chart_type}_{column_x}_{'vs_' + column_y if chart_type == 'scatter' else ''}.png"
        chart_file_path = os.path.join(user_folder, chart_file_name)

        # Remove old chart of the same type
        if os.path.exists(chart_file_path):
            os.remove(chart_file_path)

        # Generate chart and save it
        buf = BytesIO()
        fig, ax = plt.subplots(figsize=(4, 3))
        if chart_type == 'pie':
            df[column_x].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
            plt.title(f'{column_x} Distribution (Pie Chart)')
            plt.ylabel('')
        elif chart_type == 'histogram':
            df[column_x].value_counts().plot(kind='bar', ax=ax)
            plt.title(f'{column_x} Distribution (Histogram)')
            plt.ylabel('Frequency')
        elif chart_type == 'scatter':
            df.plot.scatter(x=column_x, y=column_y, ax=ax)
            plt.title(f'{column_x} vs {column_y} (Scatter Plot)')

        plt.savefig(chart_file_path, format='png')  # Save the chart to the user's folder
        plt.savefig(buf, format='png')  # Save to buffer for frontend
        plt.close()

        buf.seek(0)
        chart_base64 = base64.b64encode(buf.read()).decode('utf-8')  # Base64 encode the image for frontend display
        buf.close()

        # Save action to history
        action = Historique(
            action='Visualisation',
            date_action=datetime.now(),
            user=request.user,
            infos=f"Generated {chart_type} chart for the dataset {dataset.name}."
        )
        action.save()

        # Respond with chart path and Base64
        return JsonResponse({
            'chart_base64': chart_base64,
            'chart_url': os.path.join(settings.MEDIA_URL, 'charts', request.user.username, chart_file_name),
            'message': f'{chart_type.capitalize()} chart generated successfully!'
        })

    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)




#end visualisation 


# modeles 
@login_required(login_url='/login')
def modeles(request):
    datasets = DatasetCopy.objects.filter(user=request.user)  # Get datasets for the logged-in user
    return render(request, 'modeles.html', {'datasets': datasets})
@login_required(login_url='/login')
def get_MLcolumns(request):
    dataset_id = request.GET.get('dataset_id')  # Get dataset ID from request
    
    try:
        dataset = DatasetCopy.objects.get(id=dataset_id, user=request.user)  # Ensure dataset belongs to the user
        file_path = dataset.file.path  # Get file path of dataset
        
    
        
        # Read dataset into a DataFrame
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return JsonResponse({'error': 'Unsupported file format'}, status=400)

        columns = df.columns.tolist()  # Get list of all column names
        column_name = request.GET.get('column_name')  # Target column selected by user

        if column_name:  # Target column selected
            if column_name in df.columns:
                # Infer column type
                if pd.api.types.is_numeric_dtype(df[column_name]):
                    unique_values = df[column_name].nunique()
                    if unique_values < 20:  # Considered categorical for classification
                        column_type = 'classification'
                    else:  # Continuous for regression
                        column_type = 'regression'
                else:
                    column_type = 'classification'  # Non-numeric is always classification
                
                # Update the target and model type in DatasetCopy
                dataset.target = column_name
                dataset.type_modele = column_type
                dataset.save()
            else:
                return JsonResponse({'error': 'Column not found in dataset'}, status=400)
        else:  # No target column selected, default to clustering
            column_type = 'clustering'
            dataset.target = None  # Clear the target
            dataset.type_modele = column_type
            dataset.save()

        return JsonResponse({'columns': columns, 'column_type': column_type})

    except DatasetCopy.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def apply_models(request):
    if request.method == 'POST':
        dataset_id = request.POST.get('dataset_id')
        target_column = request.POST.get('target_column')
        models = json.loads(request.POST.get('models'))

        try:
            # Load the dataset
            dataset = DatasetCopy.objects.get(id=dataset_id, user=request.user)

    
        # Check if preprocessing is required
            preprocessing_required = (
     False  # Cleaned is false
     # Encoded is false
)
            print(f"Dataset Preprocessing Flags: normalized={dataset.status_normalized}, standardized={dataset.status_standardized}, cleaned={dataset.status_cleaned}, encoded={dataset.status_encoded}")
            print(f"Preprocessing Required: {preprocessing_required}")

            if preprocessing_required:
                return JsonResponse({'preprocessing_required': True}, status=400)
            
            file_path = dataset.file.path
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                return JsonResponse({'error': 'Unsupported file format'}, status=400)
            
            df = df.dropna()
             # Encode non-numeric columns
            for column in df.columns:
                if df[column].dtype == 'object' or df[column].dtype.name == 'category':
                    le = LabelEncoder()
                    df[column] = le.fit_transform(df[column])


            # Split data into features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Handle categorical vs. continuous targets
            is_classification = len(y.unique()) < 20  # Example threshold for categorical data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Dictionary to store metrics for each model
            metrics = []

            # Apply selected models
            for model_name in models:
                if model_name == 'KNN':
                    model = KNeighborsClassifier()
                elif model_name == 'SVM':
                    model = SVC()
                elif model_name == 'Random Forest':
                    model = RandomForestClassifier()
                elif model_name == 'Decision Tree':
                    model = DecisionTreeClassifier()
                elif model_name == 'Naive Bayes':
                    model = GaussianNB()
                elif model_name == 'Linear Regression':
                    model = LinearRegression()
                elif model_name == 'Ridge Regression':
                    model = Ridge()
                elif model_name == 'KMeans':
                    model = KMeans(n_clusters=3, random_state=42)
                elif model_name == 'DBSCAN':
                    model = DBSCAN()
                else:
                    continue

                # Train and evaluate the model
                if model_name in ['KNN', 'SVM', 'Random Forest', 'Decision Tree', 'Naive Bayes']:  # Classification
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics.append({
                        'model': model_name,
                        'accuracy': round(accuracy_score(y_test, y_pred) , 3),
                        'precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0),3),
                        'recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0),3),
                    })
                elif model_name in ['Linear Regression', 'Ridge Regression']:  # Regression
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics.append({
                        'model': model_name,
                        'mse': mean_squared_error(y_test, y_pred),
                    })
                elif model_name in ['KMeans', 'DBSCAN']:  # Clustering
                    model.fit(X)
                    if hasattr(model, 'labels_'):
                        labels = model.labels_
                        metrics.append({
                            'model': model_name,
                            'silhouette_score': silhouette_score(X, labels) if len(set(labels)) > 1 else 'N/A',
                        })
                # Log the action history
            action = Historique(
                action='Apply Models',
                date_action=datetime.now(),
                user=request.user,
                infos=f"Models applied successfully on dataset '{dataset.name}'"
            )
            action.save()
            # messages.success(request, 'Models applied successfully.')  
            message = 'Models applied successfully.'
            return JsonResponse({'metrics': metrics,'preprocessing_required': False,'message':message})

        except DatasetCopy.DoesNotExist:
            return JsonResponse({'error': 'Dataset Not Found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)



@login_required(login_url='/login')
def save_model_selection(request):
    if request.method == 'POST':
        dataset_id = request.POST.get('dataset_id')
        target_column = request.POST.get('target_column')
        model_name = request.POST.get('model_name')

        if not dataset_id or not target_column or not model_name:
            return JsonResponse({'error': 'Tous les champs sont obligatoires.'}, status=400)

        # Sauvegarder les choix dans la session
        request.session['selected_dataset_id'] = dataset_id
        request.session['selected_target_column'] = target_column
        request.session['selected_model_name'] = model_name

        return JsonResponse({'message': 'Sélection sauvegardée avec succès.'})
    return JsonResponse({'error': 'Méthode non valide.'}, status=400)
#end modeles 


# Predictions Modeles


@login_required(login_url='/login')
def predictions_page(request):
    dataset_id = request.session.get('selected_dataset_id')
    target_column = request.session.get('selected_target_column')

    if not dataset_id or not target_column:
        return render(request, 'predictionsmodeles.html', {'error': 'Sélectionnez un dataset, un target et un modèle.'})

    try:
        # Charger la dataset
        dataset = DatasetCopy.objects.get(id=dataset_id, user=request.user)
        file_path = dataset.file.path
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return render(request, 'predictionsmodeles.html', {'error': 'Format de fichier non pris en charge.'})

        # Exclure la colonne target
        input_columns = [col for col in df.columns if col != target_column]
        return render(request, 'predictionsmodeles.html', {'columns': input_columns, 'error': None})
    except DatasetCopy.DoesNotExist:
        return render(request, 'predictionsmodeles.html', {'error': 'Dataset introuvable.'})


@login_required(login_url='/login')
def Predictions(request):
    if request.method == 'POST':
        try:
            inputs = request.POST.get('inputs')
            if not inputs:
                return JsonResponse({'error': 'Aucune donnée fournie.'}, status=400)

            model_name = request.session.get('selected_model_name')
            dataset_id = request.session.get('selected_dataset_id')
            target_column = request.session.get('selected_target_column')

            if not model_name or not dataset_id or not target_column:
                return JsonResponse({'error': 'Informations manquantes.'}, status=400)

            # Charger le modèle et le dataset
            dataset = DatasetCopy.objects.get(id=dataset_id, user=request.user)
            file_path = dataset.file.path
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                return JsonResponse({'error': 'Format non pris en charge.'}, status=400)


            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Préparation des données
            input_data = json.loads(inputs)
            input_data_dict = {item['name']: float(item['value']) for item in input_data}
            input_df = pd.DataFrame([input_data_dict])

            # Charger le modèle
            trained_models = {
                'KNN': KNeighborsClassifier(),
                'SVM': SVC(),
                'Random Forest': RandomForestClassifier(),
                'Decision Tree': DecisionTreeClassifier(),
                'Naive Bayes': GaussianNB(),
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(),
                'KMeans': KMeans(n_clusters=3, random_state=42),
                'DBSCAN': DBSCAN(),
                
            }
            model = trained_models.get(model_name)
            model.fit(X, y)  # Entraîner avec les données
            prediction = model.predict(input_df)

            return JsonResponse({'prediction': prediction[0]})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Méthode non valide.'}, status=400)

#fin preditions modeles

#predictions
@login_required(login_url='/login')
def data(request):
    datasets = DatasetCopy.objects.filter(user=request.user)
    return render(request, 'predictions.html', {'datasets': datasets})

@login_required(login_url='/login')
def get_prediction_inputs(request):
    dataset_id = request.GET.get('dataset_id')
    try:
        dataset = DatasetCopy.objects.get(id=dataset_id, user=request.user)
        preprocessing_required = (
    not (dataset.status_normalized or dataset.status_standardized)  # Both normalized and standardized are false
    or not dataset.status_cleaned  # Cleaned is false
     # Encoded is false
)

        print(f"Dataset Preprocessing Flags: normalized={dataset.status_normalized}, standardized={dataset.status_standardized}, cleaned={dataset.status_cleaned}")
        print(f"Preprocessing Required: {preprocessing_required}")

        if preprocessing_required:
            return JsonResponse({'preprocessing_required': True}, status=200)

        file_path = dataset.file.path

        # Load the dataset
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return JsonResponse({'error': 'Unsupported file format'}, status=400)

        attributes = [col for col in df.columns if col != dataset.target]
        return JsonResponse({'attributes': attributes, 'type_modele': dataset.type_modele, 'preprocessing_required': False})

    except DatasetCopy.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)


@csrf_exempt
def predict(request):
    if request.method == 'POST':
        dataset_id = request.POST.get('dataset_id')
        model_name = request.POST.get('model_name')
        inputs = json.loads(request.POST.get('inputs'))

        try:
            dataset = DatasetCopy.objects.get(id=dataset_id)
            file_path = dataset.file.path

            # Load and preprocess dataset
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            df = df.dropna()

            

            # Extract target and features
            X = df.drop(columns=[dataset.target])
            y = df[dataset.target] if dataset.target else None

            # Encode non-numeric columns
            for column in X.columns:
                if X[column].dtype == 'object' or X[column].dtype.name == 'category':
                    le = LabelEncoder()
                    X[column] = le.fit_transform(X[column])

            # Prepare input data
            input_dict = {item['name']: float(item['value']) for item in inputs}
            input_df = pd.DataFrame([input_dict])

            # Train the selected model
            model_mapping = {
                'SVM': SVC(),
                'KNN': KNeighborsClassifier(),
                'Random Forest': RandomForestClassifier(),
                'Decision Tree': DecisionTreeClassifier(),
                'Naive Bayes': GaussianNB(),
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(),
                'KMeans': KMeans(n_clusters=3, random_state=42),
                'DBSCAN': DBSCAN()
            }
            model = model_mapping.get(model_name)
            if model_name in ['SVM', 'KNN', 'Random Forest', 'Decision Tree', 'Naive Bayes']:
                model.fit(X, y)
            elif model_name in ['Linear Regression', 'Ridge Regression']:
                model.fit(X, y)
            elif model_name in ['KMeans', 'DBSCAN']:
                model.fit(X)
                
            # Perform prediction
            prediction = model.predict(input_df)

            action = Historique(
                action='Apply Prediction',
                date_action=datetime.now(),
                user=request.user,
                infos=f"Prediction applied successfully on dataset '{dataset.name}'"
            )
            action.save()
            # messages.success(request, 'Prediction applied successfully.')
            message = 'Prediction applied successfully.'  
            return JsonResponse({'prediction': prediction[0],'message':message})

        except DatasetCopy.DoesNotExist:
            return JsonResponse({'error': 'Dataset not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)




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




# historique 
@login_required(login_url='/login')
# @require_GET
def historique(request):

    actions = Historique.objects.all().order_by('-id')
    context = {
        # 'selected_role': authenticated_user,
        'actions': actions,
 
    }
    return render(request, 'historique.html', context)

from django.contrib.auth.mixins import LoginRequiredMixin
from django.views import View

class DeleteHistoryView(LoginRequiredMixin, View):
    def post(self, request, *args, **kwargs):
        Historique.objects.all().delete()  # Assuming your model is named Action
        messages.success(request, 'All history records have been deleted successfully.')
        return redirect('historique')
    

import csv
# @require_POST
def export_history(request):


    actions = Historique.objects.all()



    export_format = request.POST.get('export_format')

    if export_format:
        action_history_entry = Historique(
            action='export data ',
            date_action=datetime.now(),
            user=request.user,
            infos=f"Exported data from history with format {export_format}"
        )
        action_history_entry.save()
        # messages.success(request,'History exported successfully')

        if export_format == 'csv':
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="exported_data.csv"'
            writer = csv.writer(response)
            writer.writerow(['ID Action', 'Nom Action', 'Date Action', ' Utilisateur', 'description'])
            for action in actions:
                writer.writerow(
                    [action.id, action.action, action.date_action, action.user.username +" "+ action.user.username , action.infos])
            return response
        elif export_format == 'txt':
            response = HttpResponse(content_type='text/plain')
            response['Content-Disposition'] = 'attachment; filename="exported_data.txt"'

            # Column widths
            col_widths = [15, 20, 40, 20, 50]  # Adjust the widths as needed

            # Header
            header = (
                f"{'ID Action'.ljust(col_widths[0])}"
                f"{'Nom Action'.ljust(col_widths[1])}"
                f"{'Date Action'.ljust(col_widths[2])}"
                f"{'Utilisateur'.ljust(col_widths[3])}"
                f"{'Infos'.ljust(col_widths[4])}\n"
            )

            content = header
            for action in actions:
                row = (
                    f"{str(action.id).ljust(col_widths[0])}"
                    f"{action.action.ljust(col_widths[1])}"
                    f"{str(action.date_action).ljust(col_widths[2])}"
                    f"{action.user.username.ljust(col_widths[3])}"
                    f"{action.infos.ljust(col_widths[4])}\n"
                )
                content += row

            response.write(content)
            return response
    return HttpResponse("Invalid export request", status=400)

#end historique 
