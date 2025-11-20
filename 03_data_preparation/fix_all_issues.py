import sys
from pathlib import Path

def fix_all_issues():
    """Corrige tous les problèmes identifiés"""
    print("🔧 Correction des problèmes AgriClima360...")
    
    # 1. Créer la structure de dossiers
    folders = [
        "../04_modeling/models_saved",
        "../03_data_preparation/data_noaa/processed",
        "../visualisation/massive_datashader"
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✅ Dossier créé: {folder}")
    
    # 2. Corriger les fichiers CSV problématiques
    print("\n🔧 Correction des fichiers CSV...")
    try:
        from csv_fix_processor import CSVFixProcessor
        processor = CSVFixProcessor("../03_data_preparation/data_noaa/processed")
        processor.process_all_problematic_files()
    except Exception as e:
        print(f"⚠️ Impossible de corriger les CSV: {e}")
    
    # 3. Entraîner les modèles ML
    print("\n🤖 Entraînement des modèles ML...")
    try:
        from model_regression import train_regression_model
        train_regression_model()
    except Exception as e:
        print(f"⚠️ Erreur modèle régression: {e}")
    
    try:
        from model_classification import train_classification_model
        train_classification_model()
    except Exception as e:
        print(f"⚠️ Erreur modèle classification: {e}")
    
    try:
        from model_clustering import train_clustering_model
        train_clustering_model()
    except Exception as e:
        print(f"⚠️ Erreur modèle clustering: {e}")
    
    print("\n🎉 Correction terminée! Redémarrez l'application Streamlit.")

if __name__ == "__main__":
    fix_all_issues()