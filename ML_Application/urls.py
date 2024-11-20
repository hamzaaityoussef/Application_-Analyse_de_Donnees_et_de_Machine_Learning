from django.contrib import admin
from django.urls import path , include
from ML_Application import views  # Import the views module from your app directory
from django.contrib.auth import views as auth_views


urlpatterns = [

   path('base/', views.base, name='base'),
#     # login
#     path('', views.main , name='main'),
#     path('',views.main , name='login'),
#     path('accounts/login/', views.loginpage, name='login'),
#     path('verification/', views.verification, name='verification'),

#     path('accounts/logout/', views.logoutUser, name='logout'),
#          #path('accounts/reset/', views.resetUser, name='reset'),
#     path('accounts/', include('django.contrib.auth.urls')),
#     # end login

#     #users
#     path ('users/',views.users,name="users"),
#     path('edit_user/<int:pk>/', views.edit_user, name='users1'),
#     path('delete_user/<int:id_user>/', views.delete_user, name='delete_user'),
#     path('delete_selected/<int:id_user>/', views.delete_selected, name='delete_selected'),
#       # delete entity
#     path('delete_entity/', views.delete_entity, name='delete_entity'),
#     path('get_entities/', views.get_entities, name='get_entities'),
#     # end users


#     # seeds
#     path ('seeds/',views.seeds,name='seeds'),
#     path('upload_seeds/', views.upload_seeds, name='upload_seeds'),
#     path ('export_table_data/',views.export_table_data,name='export_table_data'),

#     # path('update_seed/', views.update_seed, name='update_seed'),
#     path ('delete_seed/',views.delete_seed,name='delete_seed'),
#     # path ('upload/seeds/',views.seeds,name='upload'),
    
#     #ajax in seeds==================================
#     # URL pattern for checking if an email exists
#     # path('check_email_exists/', views.check_email_exists, name='check_email_exists'),
#     # path('fetch_rdp/', views.fetch_rdp, name='fetch_rdp'),
#     # path('fetch_session/', views.fetch_session, name='fetch_session'),
#     #end seeds

#     # actions seeds
#     path ('Actions_seeds/',views.actions_seeds,name='actions_seeds'),
#     path ('export_actions_seeds/',views.export_table_actions_seeds,name='export_table_actions_seeds'),

#     # end actions seeds



#     # 2FA actions
#     path('actions_2FA/',views.actions_2FA,name='actions_2FA'),
#     path ('export_actions_2FA/',views.export_table_actions_2FA,name='export_table_actions_2FA'),



#     #classement
#     path ('classement/',views.classement,name='classement'),
#     path ('update_local_ids/',views.update_local_ids,name='update_local_ids'),
#     path ('delete_multiple_seeds/',views.delete_multiple_seeds,name='delete_multiple_seeds'),
#     # path ('save_old_ids/',views.save_old_ids,name='save_old_ids'),
#     path ('export_data/',views.export_data,name='export_data'),
#     path ('preview_modal/',views.preview_modal,name='preview_modal'),
#     path('export_preview_to_csv/', views.export_preview_to_csv, name='export_preview_to_csv'),
#     #end classement
 

#     #old seeds
#     path ('old_seeds/',views.old_seeds,name='old_seeds'),
#     # path ('save_oldseeds/',views.save_oldseeds,name='save_oldseeds'),
#     path ('export_old_seeds/',views.export_old_seeds,name='export_old_seeds'),
#     path('delete_all_old_seeds/', views.DeleteAllOldSeedsView.as_view(), name='delete_all_old_seeds'),
#     #end old seeds



#     # gestion des proxies
#     # path('delete_selected_proxies/<int:id_proxy>/', views.delete_selected_proxies, name='delete_selected_proxies'),
#     # path('proxies/', views.proxies, name='proxies'),
#     # path('import_proxy/', views.import_proxy, name='import_proxy'),
#     #END gestion des proxies


#     #affectation des proxies
#     # path ('affectationproxy/',views.affectationproxy,name='affectationproxy'),
#             #pour fetch les informations de ip adresse (json)
#     path('get_seeds_by_ip/', views.get_seeds_by_ip, name='get_seeds_by_ip'),
#     path('repetition_des_proxies/', views.repetitionproxy, name='repetitionproxy'),
#     #end affectation des proxies


#     #gestion des RDPs
#     # path ('rdp/',views.rdp,name='rdp'),
#     # path('update_rdp/', views.update_rdp, name='update_rdp'),
#     # path('delete_rdp/', views.delete_rdp, name='delete_rdp'),
#     # path('update_session/', views.update_session, name='update_session'),
#     # path('delete_session/', views.delete_session, name='delete_session'),
    
#     #end gestion des RDPs




#     # path('entities/', views.entity_list, name='entity_list'),
#     # path('rdps/<int:entity_id>/', views.rdp_list, name='rdp_list'),
#     # path('sessions/<int:rdp_id>/', views.session_list, name='session_list'),
#     path('get_seed_count/<int:rdp_id>/', views.get_seed_count, name='get_seed_count'),

#     # history
#     path('history/', views.history, name='history'), 
#     path('export_history/', views.export_history, name='export_history'),
#     path('delete-history/', views.DeleteHistoryView.as_view(), name='delete_history'),


#      # connecthistory
#     path('connect_history/', views.connecthistory, name='connecthistory'),
#     path('export_history_connect/', views.export_history_connect, name='export_history_connect'),
#     path('delete-history_connect/', views.DeleteHistoryConnect.as_view(), name='delete_history_connect'),



#         # rdp 
#         # path('ajax/load-rdps/', views.load_rdps, name='ajax_load_rdps'), 
#         # path('get_rdps/', views.get_rdps, name='get_rdps'),  

# # check proxies
#     path('check-proxies/', views.upload_proxies, name='upload_proxies'),
#     path('test_proxies/', views.test_proxies, name='test_proxies'),


# # check emails
#     path('upload/', views.upload, name='upload'),




#  #****************dashborad************************
       
#     path('dashboard/', views.dashboard, name='dashboard'), 
#     path('api/user_roles_data/', views.user_roles_data, name='user_roles_data'), 
#     path('api/statue_connect_count__/', views.statue_connect_count__, name='statue_connect_count__'),
#     path('api/statues_connect_countall/', views.statues_connect_countall, name='statues_connect_countall'),

#     # path('api/proxy_list/', views.proxy_list, name='proxy_list'), 
    
#     # heee
#     path('get_seeds_data/', views.get_seeds_data, name='get_seeds_data'), 
#     # path('get_rdps/', views.get_rdp, name='get_rdps'), 
#     # path('get_sessions/', views.get_sessions, name='get_sessions'), 
#     # path('api/users_by_entity_chart/', views.users_by_entity_chart, name='users_by_entity_chart'), 
#     # path('api/rdp_count_by_entity/', views.rdp_count_by_entity, name='rdp_count_by_entity'), 
#     # path('api/session_count_by_rdp/', views.session_count_by_rdp, name='session_count_by_rdp'), 
#     path('api/action_count_by_name/', views.action_count_by_name, name='action_count_by_name'), 
#     path('action_counts_by_role/',  views.action_counts_by_role, name='action_counts_by_role'), 
#     path('entity_distribution/', views.entity_distribution, name='entity_distribution'), 
#     path('api/statue_2FA_count/', views.statue_2FA_count, name='statue_2FA_count'),
#     path('api/statue_connect_count/', views.statue_connect_count, name='statue_connect_count'),
#     path('api/get_statue_options/', views.get_statue_options, name='get_statue_options'),

#     #****************end dashborad************************
  
#     # routes for opening profiles with selenium
#     path('scenario_categories/', views.scenario_categories, name='scenario_categories'),
#     path('pause/', views.pause_script, name='pause'),
#     path('resume/', views.resume_script, name='resume'),
#     path('stop/', views.stop_script, name='stop'),
#     path('connect_profiles/', views.connect_profiles, name='connect_profiles'),
#     path('connect/', views.connect_view, name='connect'),
#     path('get_scheduled_profiles/', views.get_scheduled_profiles, name='get_scheduled_profiles'),
#     path('get_latest_profiles/', views.get_latest_profiles, name='get_latest_profiles'),
#     path('restart/', views.restart_script, name='restart'),
#     path('check_status/', views.check_status, name='check_status'),
#     path('check_status_profile/', views.check_status_Profile, name='check_status_profile'),
#     path('force_stop/', views.force_stop_script, name='force_stop'),
#     path('kill_chrome_profiles/', views.kill_chrome_profiles, name='kill_chrome_profiles'),
#     path('delete_scheduled_task/', views.delete_scheduled_task, name='delete_scheduled_task'),
#     path('close_and_clean/', views.Close_and_clean, name='close_and_clean'),

  
# path('sub_scenarios_log_history/', views.get_sub_scenario_action_log, name='sub_scenarios_log_history'),
# path('export/', views.export_history_connect, name='export_history_connect'),


#     # end routes for opening profiles with selenium




#   # routes generating OTP
#     path('api/get_code/', views.home_otp, name='home_otp'),
#     path('api/get_code/<str:secret_key>/', views.get_otp_code, name='get_otp_code'),

#  path('uploadexcelSchedule/', views.uploadexcelSchedule, name='uploadexcelSchedule'),
    path('accounts/logout/', views.logoutUser, name='logout'),

]