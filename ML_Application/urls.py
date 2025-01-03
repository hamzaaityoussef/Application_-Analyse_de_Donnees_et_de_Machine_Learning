from django.contrib import admin
from django.urls import path , include
from ML_Application import views  # Import the views module from your app directory
from django.contrib.auth import views as auth_views
from django.urls import re_path

urlpatterns = [

    path('', views.home, name='home'),
   path('base/', views.base, name='base'),

   
   
#     # login
    
    path('login/', views.loginpage, name='login'),
    path('accounts/logout/', views.logoutUser, name='logout'),
#     # end login 


#     upload and delete datasets
    re_path(r'^upload/?$', views.upload, name='upload'),
    path('delete/<int:dataset_id>/', views.delete_dataset, name='delete_dataset'),


    path('preprocess/', views.preprocess, name='preprocess'),
    path('apply_action/', views.apply_actions, name='apply_actions'),
    

    path('visualisation/', views.visualisation, name='visualisation'),
    path('get_columns/', views.get_columns, name='get_columns'),
    path('generate_chart/', views.generate_chart, name='generate_chart'),   # URL for fetching columns

    path('modeles/', views.modeles, name='modeles'),
    path('get_MLcolumns/', views.get_MLcolumns, name='get_MLcolumns'), 

    path('Predictions/', views.Predictions, name='Predictions'),
    path('documentation/', views.documentation, name='documentation'),
    path('report/', views.report, name='report'),
    path('generate_report/', views.generate_report, name='generate_report'),
    path('historique/', views.historique, name='historique'),
    path('export_history/', views.export_history, name='export_history'),
    path('delete-history/', views.DeleteHistoryView.as_view(), name='delete_history'),


    path('apply_models/', views.apply_models, name='apply_models'),
    path('save_model_selection/', views.save_model_selection, name='save_model_selection'),
    path('predictions_page/', views.predictions_page, name='predictions_page'),
    path('data/', views.data, name='data'),
    path('get_prediction_inputs/', views.get_prediction_inputs, name='get_prediction_inputs'),
    path('predict/', views.predict, name='predict'),



]

from django.conf import settings
from django.conf.urls.static import static

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
