from django.contrib import admin
from django.urls import path , include
from ML_Application import views  # Import the views module from your app directory
from django.contrib.auth import views as auth_views


urlpatterns = [

   path('base/', views.base, name='base'),
   
   
#     # login
    
    path('login/', views.loginpage, name='login'),
    path('accounts/logout/', views.logoutUser, name='logout'),
#     # end login

#     upload
    path('upload/', views.upload, name='upload'),


]