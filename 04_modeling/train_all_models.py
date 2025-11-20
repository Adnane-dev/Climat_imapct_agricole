#!/usr/bin/env python3
"""
Script pour entraîner tous les modèles ML d'AgriClima360
"""

import sys
from pathlib import Path

def main():
    print("🤖 ENTRAÎNEMENT DE TOUS LES MODÈLES AGRI CLIMA 360")
    print("=" * 50)
    
    try:
        # Import et entraînement du modèle de régression
        print("\n1. Modèle de Régression...")
        from model_regression import train_regression_model
        train_regression_model()
        
        # Import et entraînement du modèle de classification
        print("\n2. Modèle de Classification...")
        from model_classification import train_classification_model
        train_classification_model()
        
        # Import et entraînement du modèle de clustering
        print("\n3. Modèle de Clustering...")
        from model_clustering import train_clustering_model
        train_clustering_model()
        
        print("\n🎉 TOUS LES MODÈLES ONT ÉTÉ ENTRAÎNÉS AVEC SUCCÈS!")
        print("📁 Modèles sauvegardés dans: models_saved/")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()