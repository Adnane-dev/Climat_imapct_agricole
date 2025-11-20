# 📋 Cahier des Charges Final - AgriClima360

## 1. Contexte du Projet
**Projet** : Pipeline CRISP-DM & Visualisation Massive des Données Climatiques  
**Période** : 2000-2025  
**Volume Données** : 10M+ enregistrements NOAA GHCN  
**Objectif Principal** : Agriculture de précision et gestion des risques climatiques

## 2. Périmètre Fonctionnel

### 2.1 Phase CRISP-DM
- ✅ **Business Understanding** : Définition besoins métier
- 🔄 **Data Understanding** : Exploration données NOAA
- ⏳ **Data Preparation** : Nettoyage et feature engineering
- ⏳ **Modeling** : ML prédictif (4 modèles)
- ⏳ **Evaluation** : Validation métier et technique
- ⏳ **Deployment** : API + Dashboard Streamlit

### 2.2 Visualisation Massive
- 📊 **Datashader** : Heatmaps 10M+ points
- 🎛️ **Grafana** : Dashboard temps réel
- 📈 **Plotly** : Visualisations interactives
- 🐼 **Pandas/Matplotlib** : Analyses statistiques

## 3. Livrables Attendus

### 3.1 Livrables Techniques
- [ ] Pipeline de données modulaire
- [ ] 4 modèles ML entraînés (.pkl)
- [ ] API de prédiction FastAPI
- [ ] Dashboard Streamlit interactif
- [ ] Visualisations massives Datashader

### 3.2 Livrables Documentation
- [ ] Rapport final PDF
- [ ] Documentation API
- [ ] Guide d'utilisation
- [ ] Code source commenté

## 4. Stack Technologique

### 4.1 Core Python
```python
# Data Processing
pandas, numpy, scikit-learn, xgboost

# Visualisation  
matplotlib, seaborn, plotly, datashader

# Déploiement
streamlit, fastapi, flask