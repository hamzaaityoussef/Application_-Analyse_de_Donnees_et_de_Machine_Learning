from django.contrib import admin
from django.urls import path , include
from ML_Application import views  # Import the views module from your app directory
from django.contrib.auth import views as auth_views


urlpatterns = [

    path('', views.home, name='home'),
   path('base/', views.base, name='base'),

   
   
#     # login
    
    path('login/', views.loginpage, name='login'),
    path('accounts/logout/', views.logoutUser, name='logout'),
#     # end login

#     upload and delete datasets
    path('upload/', views.upload, name='upload'),
    path('delete/<int:dataset_id>/', views.delete_dataset, name='delete_dataset'),


    path('Preprocess/', views.Preprocess, name='Preprocess'),


    path('visualisation/', views.visualisation, name='visualisation'),
    path('get_columns/', views.get_columns, name='get_columns'),
      path('generate_chart/', views.generate_chart, name='generate_chart'),   # URL for fetching columns

    path('modeles/', views.modeles, name='modeles'),
    path('Predictions/', views.Predictions, name='Predictions'),
    path('documentation/', views.documentation, name='documentation'),
    path('report/', views.report, name='report'),
    


]

from django.conf import settings
from django.conf.urls.static import static

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
