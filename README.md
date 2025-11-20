
# 🌦️ Visualisation Multi-échelle des Tendances Climatiques (2000–2025)

### 📊 Analyse climatique pour l’agriculture de précision

**Auteurs :**

* 👨‍💻 Adnane Mahamadou Saadou
* 👩‍💻 Radhia Darghoothi

---

## 🧱 Structure finale du projet

```bash
projet_visualisation_climatique/
├── app/                               # Application Streamlit principale
│   ├── climate_data_processor.py       # Prétraitement et fusion
│   ├── merge_pivoted_data.py           # Fusion fichiers pivotés
│   ├── visualization_functions.py      # Fonctions de visualisation
│   ├── streamlit_app.py                # Interface utilisateur Streamlit
│   └── data_noaa/processed/            # Données prêtes à l’analyse
│
├── dashboard/                          # Tableau de bord Plotly/Dash
│   ├── app.py
│   ├── assets/
│   └── components/
│
├── notebooks/                          # Analyses exploratoires
│   ├── 01_pandas_manipulation.ipynb
│   ├── 02_matplotlib_specialized.ipynb
│   ├── 03_seaborn_statistical.ipynb
│   ├── 04_plotly_interactive.ipynb
│   └── 05_massive_visualization.ipynb
│
├── 01_business_understanding/          # Compréhension métier
│   ├── objectifs_smart.md
│   ├── carte_risques_climatiques.ipynb
│   └── cahier_charges_final.pdf
│
├── 02_data_understanding/              # Analyse des données brutes
│   ├── notebooks/
│   │   ├── 01_pandas_manipulation.ipynb
│   │   ├── data_profiling_report.py
│   │   └── rapport_qualite_donnees.pdf
│   └── data_noaa/raw/
│       ├── 2000.csv ... 2025.csv
│       └── ghcnd-stations.txt
│
├── 03_data_preparation/                # Préparation et fusion des datasets
│   ├── climate_data_processor.py
│   ├── merge_pivoted_data.py
│   ├── feature_engineering.py
│   ├── prepare_massive_dataset.py
│   └── data_noaa/processed/
│       ├── climate_data_pivoted_*.csv
│       ├── annual_trends.csv
│       └── massive_dataset.parquet
│
├── 04_modeling/                        # Modèles ML et statistiques
│   ├── model_regression.py
│   ├── model_classification.py
│   ├── model_clustering.py
│   ├── model_timeseries.py
│   └── models_saved/
│       ├── regression_model.pkl
│       ├── classification_model.pkl
│       └── clustering_model.pkl
│
├── 05_evaluation/                      # Évaluation des modèles
│   ├── model_evaluation.py
│   ├── validation_metier.ipynb
│   └── rapport_evaluation.pdf
│
├── 06_deployment/                      # Déploiement
│   ├── api/
│   │   ├── main.py
│   │   ├── prediction_model.py
│   │   └── requirements_api.txt
│   └── streamlit_app.py
│
├── visualisation/                      # Visualisations spécialisées
│   ├── pandas/01_pandas_manipulation.ipynb
│   ├── matplotlib/02_matplotlib_specialized.ipynb
│   ├── seaborn/03_seaborn_statistical.ipynb
│   ├── plotly/04_plotly_interactive.ipynb
│   ├── massive_datashader/
│   │   ├── notebook_datashader_massive.ipynb
│   │   ├── heatmap_massive.py
│   │   └── scatter_massive.py
│   └── dashboard_grafana/
│       ├── dashboard_grafana.json
│       └── config_grafana.yaml
│
├── data/                               # Données consolidées
│   ├── raw/ -> ../02_data_understanding/data_noaa/raw/
│   ├── processed/ -> ../03_data_preparation/data_noaa/processed/
│   └── massive/
│       ├── climate_massive.parquet
│       └── aggregated_zones.parquet
│
├── rapport/                             # Rapports académiques et PDF
│   ├── rapport_visualisation.pdf
│   └── rapport_final_agriclimavis360.pdf
│
├── docs/                                # Documentation générale
│   ├── documentation_api.md
│   └── README.md
│
├── requirements.txt                     # Dépendances Python
└── README.md                            # Documentation principale
```

---
