from django.shortcuts import render
from django.shortcuts import render, HttpResponse, redirect


# Create your views here.
def base(request):
    info = {
        'user_session': 'TestSession',
        'ip_address': '127.0.0.1',
        'current_time': '2024-08-21T14:00:00'
    }
    print('bnjrjr')

    # Call the synchronous function directly
    # send_telegram_message('Run', info)

    return render(request, 'base.html', {'user_role': 'user_role'})

from django.contrib.auth import authenticate, login, logout
from django.contrib import messages



def loginpage(request):
    if request.user.is_authenticated:
        if request.user.Role == 'senior':
            return redirect('users')
        else:
            return redirect('seeds')

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            user = None

        if user is not None and user.password == password:
            # Générer un code 2FA et l'envoyer par Telegram
            
            user.save()
            if user.chat_id:
                request.session['pending_user'] = user.id_user
                return redirect('verification')
            else:
                if user.Role in ['senior', 'admin_support', 'manager']:
                    login(request, user)
                    return redirect('users')
                    
                else:
                    login(request, user)
                    return redirect('seeds')
        else:
            messages.error(request, 'Nom d\'utilisateur ou mot de passe invalide. Veuillez réessayer.')

    return render(request, 'app/accounts/login.html')





def logoutUser(request):
    logout(request)
    return redirect('login')
