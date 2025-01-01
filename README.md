# Dataset Management and Machine Learning Application

## Overview
This is a Python and Django application for managing datasets, visualizing data, and applying machine learning models. It provides an end-to-end data science workflow.

## Features
### Dataset Management
- Import datasets in CSV, Excel, or JSON formats.
- Create new datasets directly in the application.

### Data Visualization
- Generate interactive charts.
- Explore trends and data patterns.

### Machine Learning
- Train models using:
  - Regression
  - Classification
  - Clustering
### Prediction Features
- Upload new data elements to predict outcomes using trained models.
- Visualize predictions with detailed charts.

### Reporting
- Generate and export analysis reports in PDF or Excel.

### History Management
- Track the history of dataset uploads, visualizations, and predictions.
- Export history logs for sharing or record-keeping.

---

## Installation
### Prerequisites
- Python (>= 3.8)
- Django (>= 4.0)

### Steps
1. Clone the repository:
   
- git clone https://github.com/hamzaaityoussef/Application_Analyse_de_Donnees_et_de_Machine_Learning.git

2. Create a virtual environment:

-   python -m venv env
-   source env/bin/activate  # On Windows:  
-   env\Scripts\activate

3. Install the required dependencies:
-   pip install -r requirements.txt

4.  Perform database migrations:
-   python manage.py migrate

5.  Start the development server:
-   python manage.py runserver

6.  Open your web browser and navigate to: http://127.0.0.1:8000/




# Usage
1. Upload a Dataset
-   Go to the "Dataset Management" page.
-   Click "Upload" to import datasets in CSV, Excel, or JSON format.
2. Visualize Data
-   Navigate to the "Visualization" tab.
-   Select a dataset and create interactive charts to analyze trends and patterns.
3. Train ML Models
-   Access the "ML Models" section.
-   Choose an algorithm (Regression, Classification, or Clustering).
-   Configure model parameters and train the model on the selected dataset.
4. Make Predictions
-   Visit the "Predictions" page.
-   Upload new data to generate predictions using a trained model.
5. Generate Reports
-   Navigate to the "Reports" tab.
-   Select an analysis or model to generate a detailed PDF or Excel report.
#  Deployment for Production
## Steps for Deployment
1.   Use a production-grade database like PostgreSQL:

-   Update DATABASES in the settings.py file.
2.   Install Gunicorn and configure it as a WSGI server:


-   pip install gunicorn
-   gunicorn --bind 0.0.0.0:8000 dataset_ml_app.wsgi:application
3.  Set up a reverse proxy (e.g., Nginx):

-   Configure Nginx to serve your application and handle HTTPS using Let’s Encrypt.
4.  Collect static files for production:

-   python manage.py collectstatic

5.  Secure the application using environment variables for sensitive settings like the SECRET_KEY.

# Technologies Used
-   Backend: Django, Python
-   Frontend: HTML, CSS, JavaScript, Chart.js/Plotly
-   Machine Learning: Scikit-learn, Pandas, Numpy
-   Database: SQLite (default), PostgreSQL (production)


#Future Enhancements
-   Add support for deep learning models using TensorFlow or PyTorch.
-   Implement scheduling for automated analysis tasks.
-   Provide an API for programmatic access to datasets and models.
-   Introduce user roles and permissions for better data security.

# Contributing
We welcome contributions! To contribute:

1. Fork the repository.
git clone https://github.com/hamzaaityoussef/Application_Analyse_de_Donnees_et_de_Machine_Learning.git

2. Create a new branch:
git checkout -b feature-name

3. Make your changes and commit them:
git commit -m "Add new feature"

4.  Push your changes:
git push origin feature-name

5.  Open a pull request on GitHub.

#   License
This project is licensed under the MIT License. See the LICENSE file for details.

