# 🌦️ Visualisation Multi-échelle des Tendances Climatiques (2000–2025)
### 📊 Analyse climatique pour l’agriculture de précision

**Auteurs :**
- 👨‍💻 Adnane Mahamadou Saadou  
- 👩‍💻 Radhia Darghoothi  

---

## 🧠 Contexte du projet
Ce projet vise à **analyser, fusionner et visualiser les données climatiques (NOAA GHCN Daily)** sur la période **2000–2025**, dans le but d’identifier les tendances clés affectant **l’agriculture et la durabilité environnementale**.

Les objectifs principaux :
- Prétraiter et fusionner les fichiers climatologiques multi-annuels.  
- Créer des **visualisations statistiques et interactives** (Matplotlib, Seaborn, Plotly).  
- Développer un **tableau de bord Streamlit** pour l’exploration dynamique.  
- Générer un **rapport analytique automatisé** en PDF.

---

## 🧱 Structure du projet

```bash
projet_visualisation_climatique/
├── app/                         # Application Streamlit principale
│   ├── climate_data_processor.py # Prétraitement et fusion
│   ├── merge_pivoted_data.py     # Fusion des fichiers pivotés
│   ├── visualization_functions.py# Fonctions de visualisation
│   ├── streamlit_app.py          # Interface utilisateur Streamlit
│   └── data_noaa/processed/      # Données prêtes à l’analyse
│
├── dashboard/                   # Tableau de bord Plotly/Dash
│   ├── app.py
│   ├── assets/
│   └── components/
│
├── data_noaa/                   # Données NOAA brutes (2000–2025)
│   ├── 2000.csv ... 2025.csv
│   ├── ghcnd-stations.txt
│   └── processed/
│       ├── climate_data_pivoted_*.csv
│       └── annual_trends.csv
│
├── notebooks/                   # Analyse exploratoire
│   ├── 01_pandas_manipulation.ipynb
│   ├── 02_matplotlib_specialized.ipynb
│   ├── 03_seaborn_statistical.ipynb
│   ├── 04_plotly_interactive.ipynb
│   └── 05_massive_visualization.ipynb
│
├── rapport/                     # Rapport académique PDF
│   └── rapport_visualisation.pdf
│
├── requirements.txt             # Dépendances Python
└── README.md                    # Documentation principale
