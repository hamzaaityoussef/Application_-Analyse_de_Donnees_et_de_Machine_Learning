
# Create your models here.
import os
import uuid
from datetime import datetime
from django.db import models
from django.contrib.auth.models import AbstractUser

# Function to generate unique file paths for each user and date
def generate_unique_filename(instance, filename):
    """
    Génère un chemin unique pour chaque fichier, organisé par utilisateur et date d'importation.
    Exemple : /datasets/user_{id_utilisateur}/2024-11-20/uuid_nomfichier.csv
    """
    ext = os.path.splitext(filename)[1]  # Extension du fichier
    date_str = datetime.now().strftime('%Y-%m-%d')  # Date au format AAAA-MM-JJ
    return f"datasets/user_{instance.user.id}/{date_str}/{uuid.uuid4().hex}{ext}"



class User(AbstractUser):  
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255)


   
    is_active = True
    is_superuser = None
    is_staff = None

    def _str_(self):
        return f"{self.username} ({self.email})"

    class Meta:
        db_table = 'User'  


# Dataset model
class Dataset(models.Model):
    name = models.CharField(max_length=255)  # Nom du dataset
    file = models.FileField(upload_to=generate_unique_filename)  # Sauvegarde du fichier dans un chemin unique
    date_import = models.DateTimeField(auto_now_add=True)  # Date d'importation (automatique)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='datasets')  # FK vers User

    status_normalized = models.BooleanField(default=False)  # Status de normalisation
    status_standarized = models.BooleanField(default=False)  # Status de standardisation
    status_cleaned = models.BooleanField(default=False)  # Status de nettoyage
    status_encoded = models.BooleanField(default=False)  # Status d'encodage
    copied = models.BooleanField(default=False)  # Indicateur de copie

    def _str_(self):
        return self.name

    class Meta:
        db_table = 'Dataset'  # Définit le nom de la table


# Historique model
class Historique(models.Model):
    ACTION_CHOICES = [
        ('IMPORT', 'Importation'),
        ('ANALYSE', 'Analyse'),
        ('VISUALISATION', 'Visualisation'),
        ('PREPROCESSING', 'Prétraitement'),
    ]

    action = models.CharField(max_length=50, choices=ACTION_CHOICES)  # Type d'action choisie
    date_action = models.DateTimeField(auto_now_add=True)  # Date de l'action (automatique)
    dataset = models.ForeignKey(Dataset, on_delete=models.SET_NULL, null=True, related_name='historiques')  # FK vers Dataset
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='historiques')  # FK vers User
    report_or_path = models.TextField(null=True, blank=True)  # Chemin vers le rapport ou description (optionnel)

    def _str_(self):
        return f"Action: {self.get_action_display()} - Dataset: {self.dataset.name if self.dataset else 'Aucun'}"

    class Meta:
        db_table = 'Historique'  # Définit le nom de la table