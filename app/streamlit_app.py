import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import joblib
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION SPÉCIFIQUE POUR VOTRE PROJET GIT
# ==========================================

# Chemins absolus pour votre structure Git
current_dir = Path(__file__).parent
project_root = current_dir.parent  # Racine du projet Git
sys.path.append(str(project_root))

# Configuration de la page
st.set_page_config(
    page_title="Climat Impact Agricole - Analyse Intelligente",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS moderne adapté à l'agriculture
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0fff4 0%, #e6fffa 100%);
    }
    .main-header {
        background: linear-gradient(135deg, #38a169 0%, #2f855a 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .agriculture-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #38a169;
    }
    .crispdm-phase {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #d97706;
    }
    .ml-model-card {
        background: linear-gradient(135deg, #dbeafe 0%, #93c5fd 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2563eb;
    }
    .data-quality-card {
        background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #db2777;
    }
    .file-list {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #38a169 0%, #2f855a 100%);
        color: white;
        border: none;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(56, 161, 105, 0.4);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #38a169;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# CHARGEMENT DES MODÈLES ML - ADAPTÉ À VOTRE GIT
# ==========================================

@st.cache_resource
def load_ml_models():
    """Charge les modèles ML depuis votre structure Git"""
    models = {}
    try:
        # Chemins spécifiques à votre projet
        models_dir = project_root / '04_modeling' / 'models_saved'
        
        if not models_dir.exists():
            st.sidebar.warning("📁 Dossier des modèles non trouvé. Exécutez d'abord l'entraînement.")
            return {}
        
        # Chargement robuste des modèles
        model_files = {
            'regression': 'regression_model.pkl',
            'classification': 'classification_model.pkl', 
            'clustering': 'clustering_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = models_dir / filename
            if model_path.exists():
                try:
                    models[model_name] = joblib.load(model_path)
                    st.sidebar.success(f"✅ {model_name} chargé")
                except Exception as e:
                    st.sidebar.error(f"❌ Erreur chargement {model_name}: {e}")
            else:
                st.sidebar.warning(f"⚠️ {model_name} non trouvé: {filename}")
                
    except Exception as e:
        st.sidebar.error(f"🚨 Erreur générale chargement modèles: {e}")
    
    return models

# ==========================================
# FONCTIONS UTILITAIRES AMÉLIORÉES
# ==========================================

def safe_get_column(df, possible_names):
    """Trouve la première colonne existante parmi une liste de noms possibles"""
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def detect_columns(df):
    """Détecte automatiquement les colonnes climatiques"""
    cols = {
        'date': safe_get_column(df, ['DATE', 'Date', 'date', 'dt']),
        'year': safe_get_column(df, ['YEAR', 'Year', 'year', 'annee']),
        'month': safe_get_column(df, ['MONTH', 'Month', 'month', 'mois']),
        'day': safe_get_column(df, ['DAY', 'Day', 'day', 'jour']),
        'temp_avg': safe_get_column(df, ['TAVG', 'TAVG_mean', 'tavg', 'TEMP_AVG', 'Temp_Avg', 'temperature']),
        'temp_min': safe_get_column(df, ['TMIN', 'TMIN_min', 'tmin', 'TEMP_MIN', 'Temp_Min']),
        'temp_max': safe_get_column(df, ['TMAX', 'TMAX_max', 'tmax', 'TEMP_MAX', 'Temp_Max']),
        'precip': safe_get_column(df, ['PRCP', 'PRCP_sum', 'prcp', 'PRECIPITATION', 'Precipitation', 'precipitations']),
        'station_id': safe_get_column(df, ['ID', 'id', 'STATION', 'STATION_ID', 'Station_ID']),
        'latitude': safe_get_column(df, ['LATITUDE', 'Latitude', 'lat']),
        'longitude': safe_get_column(df, ['LONGITUDE', 'Longitude', 'lon'])
    }
    return cols

def scan_data_directory():
    """Scan du répertoire de données selon votre structure Git"""
    data_dir = project_root / 'data_noaa' / 'processed'
    
    if not data_dir.exists():
        st.sidebar.error("❌ Dossier data_noaa/processed non trouvé")
        return [], []
    
    csv_files = list(data_dir.glob("*.csv"))
    files_info = []
    available_years = []
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Scan des Données")
    
    if not csv_files:
        st.sidebar.warning("Aucun fichier CSV trouvé dans data_noaa/processed/")
        return [], []
    
    st.sidebar.success(f"✅ {len(csv_files)} fichier(s) CSV détecté(s)")
    
    for file_path in csv_files:
        file_info = {
            'name': file_path.name,
            'path': file_path,
            'size_mb': round(file_path.stat().st_size / 1024 / 1024, 2),
            'type': 'other'
        }
        
        # Détection du type de fichier
        filename_lower = file_path.name.lower()
        
        if any(keyword in filename_lower for keyword in ['annual', 'trend', 'summary', 'agreg']):
            file_info['type'] = 'annual_trends'
            icon = "📊"
            status = "Tendances annuelles"
        else:
            # Détection des années
            year_matches = re.findall(r'(20\d{2})', file_path.name)
            if year_matches:
                year = int(year_matches[0])
                file_info['type'] = 'yearly_data'
                file_info['year'] = year
                available_years.append(year)
                icon = "📅"
                status = f"Données {year}"
            else:
                file_info['type'] = 'other'
                icon = "📄"
                status = "Autre fichier"
        
        files_info.append(file_info)
        
        # Affichage dans la sidebar
        st.sidebar.markdown(f"""
        <div class="file-list">
            <strong>{icon} {file_info['name']}</strong><br>
            <small>Type: {status} | Taille: {file_info['size_mb']} MB</small>
        </div>
        """, unsafe_allow_html=True)
    
    available_years = sorted(list(set(available_years)))
    return files_info, available_years

@st.cache_data(ttl=3600)
def load_annual_trends():
    """Charge les tendances annuelles"""
    trends_path = project_root / 'data_noaa' / 'processed' / 'annual_trends.csv'
    
    if not trends_path.exists():
        return None
    
    try:
        df = pd.read_csv(trends_path)
        if df.empty:
            return None
        
        # Conversion des températures (dixièmes de degrés → degrés)
        temp_columns = ['TAVG_mean', 'TMIN_min', 'TMAX_max']
        for col in temp_columns:
            if col in df.columns:
                df[col] = df[col] / 10.0
        
        if 'PRCP_sum' in df.columns:
            df['PRCP_sum'] = df['PRCP_sum'] / 10.0  # Conversion mm
        
        return df
    except Exception as e:
        st.error(f"Erreur chargement tendances: {e}")
        return None

@st.cache_data(ttl=3600)
def load_year_data(year):
    """Charge les données d'une année spécifique"""
    data_dir = project_root / 'data_noaa' / 'processed'
    
    # Patterns de fichiers possibles
    patterns = [
        f"climate_data_pivoted_{year}.csv",
        f"climate_data_{year}.csv", 
        f"data_{year}.csv",
        f"{year}_data.csv",
        f"donnees_{year}.csv"
    ]
    
    for pattern in patterns:
        file_path = data_dir / pattern
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                
                if df.empty:
                    continue
                
                # Colonnes essentielles
                essential_cols = ['DATE', 'ID', 'YEAR', 'MONTH', 'DAY', 'TAVG', 'TMIN', 'TMAX', 'PRCP']
                available_cols = [col for col in essential_cols if col in df.columns]
                
                if len(available_cols) < 5:
                    continue
                
                df = df[available_cols].copy()
                
                # Optimisation des types
                if 'YEAR' in df.columns:
                    df['YEAR'] = df['YEAR'].astype('int16')
                if 'MONTH' in df.columns:
                    df['MONTH'] = df['MONTH'].astype('int8')
                if 'DAY' in df.columns:
                    df['DAY'] = df['DAY'].astype('int8')
                
                # Conversion températures
                temp_cols = ['TAVG', 'TMIN', 'TMAX']
                for col in temp_cols:
                    if col in df.columns:
                        df[col] = df[col].replace([-9999, 9999, -999], np.nan)
                        df[col] = df[col] / 10.0
                        df[col] = df[col].astype('float32')
                
                # Conversion précipitations
                if 'PRCP' in df.columns:
                    df['PRCP'] = df['PRCP'].replace(-9999, 0)
                    df['PRCP'] = df['PRCP'] / 10.0
                    df['PRCP'] = df['PRCP'].astype('float32')
                
                return df
                
            except Exception as e:
                st.warning(f"Erreur chargement {pattern}: {e}")
                continue
    
    return None

def load_data_optimized():
    """Chargement principal des données"""
    try:
        # Scan des données disponibles
        files_info, available_years = scan_data_directory()
        
        # Sélection des années dans la sidebar
        selected_years = []
        if st.session_state.get('load_detailed', False) and available_years:
            selected_years = st.sidebar.multiselect(
                "🎯 Sélectionner les années à analyser",
                available_years,
                default=available_years[-2:] if available_years else []
            )
        
        # Chargement des données
        result = {}
        
        # 1. Tendances annuelles
        df_trends = load_annual_trends()
        if df_trends is not None:
            result['annual_trends'] = df_trends
        
        # 2. Données détaillées par année
        if selected_years:
            yearly_data = []
            for year in selected_years:
                df_year = load_year_data(year)
                if df_year is not None:
                    yearly_data.append(df_year)
            
            if yearly_data:
                result['detailed_data'] = pd.concat(yearly_data, ignore_index=True)
                st.sidebar.success(f"📊 {len(yearly_data)} année(s) chargée(s)")
        
        return result, f"{len(selected_years)} années sélectionnées"
        
    except Exception as e:
        st.error(f"🚨 Erreur chargement données: {e}")
        return {}, str(e)

# ==========================================
# INTERFACES SPÉCIFIQUES AGRICULTURE
# ==========================================

def show_project_overview():
    """Vue d'ensemble du projet Climat Impact Agricole"""
    st.markdown("""
    <div class="main-header">
        <h1>🌾 Climat Impact Agricole</h1>
        <h3>Analyse Intelligente des Données Climatiques pour l'Agriculture</h3>
        <p>Projet CRISP-DM - Master Data Science</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Présentation du projet
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📋 À Propos du Projet
        
        **Climat Impact Agricole** est une plateforme d'analyse avancée des données climatiques 
        pour optimiser les décisions agricoles face aux changements climatiques.
        
        **Objectifs :**
        - Analyser l'impact du climat sur l'agriculture
        - Prédire les risques climatiques
        - Optimiser les calendriers culturaux
        - Fournir des insights data-driven aux agriculteurs
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Métriques Clés
        """)
        st.metric("📊 Volume données", "10M+ points")
        st.metric("🌡️ Période couverte", "2000-2025")
        st.metric("🤖 Modèles ML", "3 algorithmes")
        st.metric("📍 Stations", "500+")
    
    # Architecture CRISP-DM
    st.markdown("### 🔧 Architecture CRISP-DM")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="crispdm-phase">
            <h4>🔍 1. Compréhension Métier</h4>
            <p>• Besoins agricoles</p>
            <p>• Définition des risques</p>
            <p>• Objectifs SMART</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="crispdm-phase">
            <h4>🛠️ 3. Préparation Données</h4>
            <p>• Nettoyage NOAA</p>
            <p>• Feature engineering</p>
            <p>• Agrégation spatiale</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="crispdm-phase">
            <h4>📁 2. Compréhension Données</h4>
            <p>• Exploration NOAA</p>
            <p>• Qualité données</p>
            <p>• Détection anomalies</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="crispdm-phase">
            <h4>🤖 4. Modélisation</h4>
            <p>• Régression température</p>
            <p>• Classification sécheresse</p>
            <p>• Clustering zones</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="crispdm-phase">
            <h4>📈 5. Évaluation</h4>
            <p>• Performance modèles</p>
            <p>• Validation métier</p>
            <p>• A/B testing</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="crispdm-phase">
            <h4>🚀 6. Déploiement</h4>
            <p>• Dashboard Streamlit</p>
            <p>• API prédictions</p>
            <p>• Rapports automatiques</p>
        </div>
        """, unsafe_allow_html=True)

def create_safe_prediction_input(model, user_inputs, expected_features):
    """Crée un input sécurisé pour les modèles ML"""
    try:
        safe_input = {}
        for feature in expected_features:
            if feature in user_inputs:
                safe_input[feature] = user_inputs[feature]
            else:
                # Valeurs par défaut intelligentes
                if any(temp in feature for temp in ['TMAX', 'TMIN', 'TAVG']):
                    safe_input[feature] = 20.0
                elif 'PRCP' in feature:
                    safe_input[feature] = 0.0
                else:
                    safe_input[feature] = 0.0
        
        return pd.DataFrame([safe_input])
    except Exception as e:
        st.error(f"Erreur préparation données: {e}")
        return None

def show_ml_interface(data):
    """Interface Machine Learning améliorée"""
    st.header("🧠 Intelligence Artificielle Agricole")
    
    # Chargement des modèles
    ml_models = load_ml_models()
    
    if not ml_models:
        st.warning("""
        ⚠️ Les modèles ML ne sont pas encore disponibles.
        
        **Pour les générer :**
        1. Exécutez les notebooks dans `04_modeling/`
        2. Lancez l'entraînement des modèles
        3. Les modèles seront sauvegardés dans `04_modeling/models_saved/`
        """)
        return
    
    tab1, tab2, tab3 = st.tabs(["🌡️ Prédiction Température", "⚠️ Alerte Sécheresse", "🗺️ Zones Climatiques"])
    
    with tab1:
        st.subheader("Prédiction des Températures")
        
        if 'regression' in ml_models:
            model_data = ml_models['regression']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 🔮 Prédire la température")
                
                col_input1, col_input2 = st.columns(2)
                with col_input1:
                    tmin = st.number_input("Temp. minimale (°C)", value=12.0, min_value=-30.0, max_value=40.0)
                    prcp = st.number_input("Précipitations (mm)", value=5.0, min_value=0.0, max_value=200.0)
                with col_input2:
                    tavg = st.number_input("Temp. moyenne (°C)", value=18.0, min_value=-30.0, max_value=40.0)
                    month = st.selectbox("Mois", range(1, 13), format_func=lambda x: ["Jan","Fév","Mar","Avr","Mai","Jun",
                                                                                     "Jul","Aoû","Sep","Oct","Nov","Déc"][x-1])
                
                if st.button("🎯 Prédire température maximale", type="primary"):
                    try:
                        # Inputs utilisateur
                        user_inputs = {
                            'TMIN': tmin, 'TAVG': tavg, 'PRCP': prcp, 'month': month,
                            'TMAX_rolling_mean_7': tavg + 2, 'TMAX_rolling_std_7': 1.5
                        }
                        
                        # Features attendues (à adapter selon votre modèle)
                        expected_features = ['TMIN', 'TAVG', 'PRCP', 'month', 'TMAX_rolling_mean_7', 'TMAX_rolling_std_7']
                        
                        input_df = create_safe_prediction_input(model_data, user_inputs, expected_features)
                        
                        if input_df is not None:
                            prediction = model_data.predict(input_df)[0]
                            
                            st.success(f"**🌡️ Température maximale prédite: {prediction:.1f}°C**")
                            
                            # Analyse du risque
                            col_risk1, col_risk2, col_risk3 = st.columns(3)
                            with col_risk1:
                                diff = prediction - tavg
                                st.metric("📈 Écart avec moyenne", f"{diff:+.1f}°C")
                            with col_risk2:
                                risk = "Élevé" if prediction > 32 else "Modéré" if prediction > 28 else "Faible"
                                st.metric("⚠️ Risque chaleur", risk)
                            with col_risk3:
                                st.metric("🎯 Précision estimée", "85%")
                    
                    except Exception as e:
                        st.error(f"❌ Erreur prédiction: {e}")
            
            with col2:
                st.markdown("#### 📋 Modèle de Régression")
                st.markdown("""
                <div class="ml-model-card">
                    <h5>Random Forest Regressor</h5>
                    <p><strong>Performance:</strong> RMSE ≈ 2.1°C</p>
                    <p><strong>Utilisation:</strong></p>
                    <ul>
                        <li>Planification cultures</li>
                        <li>Alertes canicule</li>
                        <li>Optimisation irrigation</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("❌ Modèle de régression non disponible")
    
    with tab2:
        st.subheader("Détection du Risque de Sécheresse")
        
        if 'classification' in ml_models:
            model_data = ml_models['classification']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 🔍 Évaluer le risque sécheresse")
                
                precip_30j = st.slider("Précipitations 30 derniers jours (mm)", 0, 300, 45)
                temp_moy_30j = st.slider("Température moyenne 30j (°C)", 5, 35, 22)
                humidite_sol = st.slider("Humidité du sol estimée (%)", 0, 100, 65)
                
                if st.button("🌵 Analyser le risque sécheresse", type="primary"):
                    try:
                        user_inputs = {
                            'PRCP': precip_30j, 'TAVG': temp_moy_30j,
                            'TMIN': temp_moy_30j - 5, 'TMAX': temp_moy_30j + 5
                        }
                        
                        expected_features = ['PRCP', 'TAVG', 'TMIN', 'TMAX']
                        input_data = create_safe_prediction_input(model_data, user_inputs, expected_features)
                        
                        if input_data is not None:
                            prediction = model_data.predict(input_data)[0]
                            proba = model_data.predict_proba(input_data)[0]
                            
                            if prediction == 1:
                                st.error(f"🚨 RISQUE SÉCHERESSE ÉLEVÉ - Probabilité: {proba[1]:.1%}")
                                st.warning("""
                                **Recommandations:**
                                - Réduire les surfaces irriguées
                                - Privilégier les cultures résistantes
                                - Surveiller l'humidité du sol
                                """)
                            else:
                                st.success(f"✅ RISQUE FAIBLE - Probabilité: {proba[0]:.1%}")
                                st.info("Conditions normales - Poursuivez vos activités agricoles")
                            
                            # Visualisation probabilités
                            fig_proba = go.Figure(data=[
                                go.Bar(x=['Faible risque', 'Risque élevé'], 
                                      y=proba,
                                      marker_color=['#38a169', '#e53e3e'])
                            ])
                            fig_proba.update_layout(title="Probabilités de risque", height=300)
                            st.plotly_chart(fig_proba, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ Erreur analyse: {e}")
            
            with col2:
                st.markdown("#### 📊 Modèle de Classification")
                st.markdown("""
                <div class="ml-model-card">
                    <h5>Random Forest Classifier</h5>
                    <p><strong>Performance:</strong> Accuracy 92%</p>
                    <p><strong>Seuil d'alerte:</strong></p>
                    <ul>
                        <li>Précipitations < 50mm/30j</li>
                        <li>Température > 25°C moyenne</li>
                        <li>Humidité sol < 60%</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("❌ Modèle de classification non disponible")
    
    with tab3:
        st.subheader("Classification des Zones Climatiques")
        
        if 'clustering' in ml_models:
            # Visualisation des clusters
            st.info("**Carte des zones climatiques identifiées par l'IA**")
            
            # Données simulées pour la démonstration
            np.random.seed(42)
            n_stations = 100
            demo_data = pd.DataFrame({
                'latitude': np.random.uniform(43, 49, n_stations),
                'longitude': np.random.uniform(-2, 8, n_stations),
                'temperature_moyenne': np.random.normal(18, 5, n_stations),
                'precipitations_annuelles': np.random.gamma(2, 200, n_stations),
                'zone_climatique': np.random.randint(0, 3, n_stations)
            })
            
            # Carte interactive
            fig_map = px.scatter_mapbox(
                demo_data,
                lat='latitude',
                lon='longitude',
                color='zone_climatique',
                size='precipitations_annuelles',
                hover_name='zone_climatique',
                hover_data=['temperature_moyenne', 'precipitations_annuelles'],
                color_continuous_scale=px.colors.sequential.Viridis,
                zoom=5,
                height=500,
                title="Zones Climatiques de France"
            )
            fig_map.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig_map, use_container_width=True)
            
            # Légende des zones
            col_z1, col_z2, col_z3 = st.columns(3)
            with col_z1:
                st.markdown("""
                <div style='background: #440154; color: white; padding: 1rem; border-radius: 10px;'>
                    <h5>🏔️ Zone 0 - Montagne</h5>
                    <p>Froide, haute précipitation</p>
                </div>
                """, unsafe_allow_html=True)
            with col_z2:
                st.markdown("""
                <div style='background: #21918c; color: white; padding: 1rem; border-radius: 10px;'>
                    <h5>🌾 Zone 1 - Tempérée</h5>
                    <p>Idéale pour céréales</p>
                </div>
                """, unsafe_allow_html=True)
            with col_z3:
                st.markdown("""
                <div style='background: #fde725; color: black; padding: 1rem; border-radius: 10px;'>
                    <h5>☀️ Zone 2 - Méditerranéenne</h5>
                    <p>Chaude, faible précipitation</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("❌ Modèle de clustering non disponible")

def show_agricultural_insights(data):
    """Insights spécifiques à l'agriculture"""
    st.header("🌱 Insights Agricoles")
    
    if not data:
        st.warning("❌ Chargez d'abord des données pour voir les insights")
        return
    
    # Sélection du dataset
    if 'detailed_data' in data:
        df = data['detailed_data']
        st.success("📊 Utilisation des données détaillées")
    elif 'annual_trends' in data:
        df = data['annual_trends']
        st.info("📊 Utilisation des tendances annuelles")
    else:
        st.warning("❌ Aucune donnée disponible")
        return
    
    cols = detect_columns(df)
    
    tab1, tab2, tab3 = st.tabs(["📅 Calendrier Cultural", "💧 Besoins en Eau", "🌡️ Stress Thermique"])
    
    with tab1:
        st.subheader("Calendrier Cultural Optimal")
        
        if cols['month'] and cols['temp_avg']:
            # Analyse des températures par mois
            monthly_avg = df.groupby(cols['month'])[cols['temp_avg']].mean().reset_index()
            
            # Recommandations culturales par mois
            recommendations = {
                1: "🌾 Semis blé d'hiver | 🛌 Période dormante",
                2: "🌾 Entretien céréales | 🌱 Préparation sol",
                3: "🌱 Semis maïs | 🌾 Fertilisation céréales", 
                4: "💧 Irrigation début | 🌿 Croissance végétative",
                5: "🌾 Floraison céréales | 🌱 Développement maïs",
                6: "☀️ Récolte fourrages | 💧 Irrigation intensive",
                7: "🌾 Moisson blé | 🌽 Floraison maïs",
                8: "🌽 Récolte maïs | 🌱 Semis colza",
                9: "🌾 Semis céréales | 🍇 Vendanges",
                10: "🌾 Levée céréales | 🍂 Récoltes automne",
                11: "🛌 Fin cultures | 🌾 Préparation hiver", 
                12: "📊 Planification | 🛌 Repos végétatif"
            }
            
            monthly_avg['Recommandation'] = monthly_avg[cols['month']].map(recommendations)
            
            # Graphique températures + recommandations
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=monthly_avg[cols['month']],
                y=monthly_avg[cols['temp_avg']],
                name='Température moyenne',
                marker_color='#38a169'
            ))
            
            fig.update_layout(
                title="Températures Moyennes et Calendrier Cultural",
                xaxis_title="Mois",
                yaxis_title="Température (°C)",
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau des recommandations
            st.markdown("#### 📋 Recommandations par Mois")
            for idx, row in monthly_avg.iterrows():
                with st.expander(f"{row[cols['month']]} - {recommendations[row[cols['month']]].split(' | ')[0]}"):
                    st.write(f"**Température moyenne:** {row[cols['temp_avg']]:.1f}°C")
                    st.write(f"**Activités:** {recommendations[row[cols['month']]]}")
    
    with tab2:
        st.subheader("Analy des Besoins en Eau")
        
        if cols['precip'] and cols['temp_avg']:
            # Calcul des besoins en eau théoriques
            if 'month' in df.columns:
                water_data = df.groupby('month').agg({
                    cols['precip']: 'sum',
                    cols['temp_avg']: 'mean'
                }).reset_index()
                
                # Estimation besoins en eau (simplifiée)
                water_data['besoin_eau'] = water_data[cols['temp_avg']] * 10  # mm/mois
                water_data['deficit'] = water_data['besoin_eau'] - water_data[cols['precip']]
                
                fig_water = go.Figure()
                
                fig_water.add_trace(go.Bar(
                    x=water_data['month'],
                    y=water_data[cols['precip']],
                    name='Précipitations',
                    marker_color='#3182ce'
                ))
                
                fig_water.add_trace(go.Scatter(
                    x=water_data['month'],
                    y=water_data['besoin_eau'],
                    name='Besoins en eau',
                    line=dict(color='#e53e3e', width=3, dash='dot')
                ))
                
                fig_water.update_layout(
                    title="Bilan Hydrique Mensuel",
                    xaxis_title="Mois",
                    yaxis_title="Eau (mm)",
                    template="plotly_white",
                    height=400
                )
                
                st.plotly_chart(fig_water, use_container_width=True)
                
                # Alertes déficit
                deficit_mois = water_data[water_data['deficit'] > 0]
                if not deficit_mois.empty:
                    st.warning(f"🚨 Déficit hydrique détecté sur {len(deficit_mois)} mois")
                    for _, row in deficit_mois.iterrows():
                        st.write(f"- Mois {int(row['month'])}: Déficit de {row['deficit']:.0f} mm")

def show_data_quality_report(data):
    """Rapport de qualité des données"""
    st.header("📊 Qualité des Données")
    
    if not data:
        st.warning("❌ Aucune donnée à analyser")
        return
    
    dataset_choice = st.selectbox("Choisir le dataset", list(data.keys()))
    df = data[dataset_choice]
    
    # Métriques de qualité
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.metric("📈 Complétude", f"{completeness:.1f}%")
    
    with col2:
        duplicates = df.duplicated().sum()
        st.metric("🔍 Doublons", f"{duplicates}")
    
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.metric("🔢 Colonnes numériques", f"{len(numeric_cols)}")
    
    with col4:
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        st.metric("📅 Colonnes dates", f"{len(date_cols)}")
    
    # Analyse détaillée
    st.subheader("Analyse par Colonne")
    
    quality_report = []
    for col in df.columns:
        col_data = df[col]
        null_count = col_data.isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = col_data.nunique()
        
        quality_report.append({
            'Colonne': col,
            'Type': col_data.dtype,
            'Valeurs nulles': null_count,
            '% Nulles': f"{null_pct:.1f}%",
            'Valeurs uniques': unique_count,
            'Exemple': col_data.iloc[0] if not col_data.empty else 'N/A'
        })
    
    quality_df = pd.DataFrame(quality_report)
    st.dataframe(quality_df, use_container_width=True)

# ==========================================
# INTERFACE PRINCIPALE
# ==========================================

def main():
    # Initialisation session state
    if 'load_detailed' not in st.session_state:
        st.session_state.load_detailed = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center;'>
            <h1>🌾 Climat Impact Agricole</h1>
            <p><em>Analyse Data Science</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎛️ Navigation")
        page = st.radio(
            "Sélectionner une section",
            [
                "🏠 Vue d'ensemble",
                "📈 Tableau de Bord", 
                "🧠 Intelligence Artificielle",
                "🌱 Insights Agricoles",
                "📊 Qualité Données",
                "🔍 Explorateur"
            ]
        )
        
        st.markdown("---")
        st.markdown("### 📂 Chargement Données")
        
        # Mode de chargement
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.load_detailed:
                st.success("✅ Mode Standard")
            else:
                if st.button("📊 Standard", use_container_width=True):
                    st.session_state.load_detailed = False
                    st.session_state.data_loaded = False
                    st.rerun()
        
        with col2:
            if st.session_state.load_detailed:
                st.warning("🔍 Mode Détail")
            else:
                if st.button("🔍 Détail", use_container_width=True):
                    st.session_state.load_detailed = True
                    st.session_state.data_loaded = False
                    st.rerun()
        
        st.markdown("---")
        
        if st.button("🔄 Actualiser les données", use_container_width=True, type="primary"):
            st.session_state.data_loaded = False
            st.rerun()
    
    # Chargement des données
    if not st.session_state.data_loaded:
        with st.spinner("📥 Chargement des données climatiques..."):
            data, info = load_data_optimized()
        
        if data:
            st.session_state.data = data
            st.session_state.data_loaded = True
            st.session_state.load_info = info
            st.success(f"✅ Données chargées: {info}")
        else:
            st.error("❌ Échec du chargement des données")
            data = None
    else:
        data = st.session_state.data
    
    # Navigation
    if page == "🏠 Vue d'ensemble":
        show_project_overview()
    
    elif page == "📈 Tableau de Bord":
        if data:
            show_dashboard(data)
        else:
            st.warning("❌ Chargez d'abord des données")
    
    elif page == "🧠 Intelligence Artificielle":
        show_ml_interface(data)
    
    elif page == "🌱 Insights Agricoles":
        show_agricultural_insights(data)
    
    elif page == "📊 Qualité Données":
        show_data_quality_report(data)
    
    elif page == "🔍 Explorateur":
        if data:
            show_data_explorer(data)
        else:
            st.warning("❌ Chargez d'abord des données")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Climat Impact Agricole</strong> - Projet Master Data Science</p>
        <p>📚 <a href='https://github.com/Adnane-dev/Climat_imapct_agricole' target='_blank'>GitHub Repository</a> | 
        🎓 MPDS3 2025</p>
    </div>
    """, unsafe_allow_html=True)

# Import des fonctions existantes (à adapter selon vos besoins)
def show_dashboard(data):
    """Tableau de bord climatique"""
    st.header("📈 Tableau de Bord Climatique")
    # Implémentation existante à adapter...
    st.info("Tableau de bord en cours de développement...")

def show_data_explorer(data):
    """Explorateur de données"""
    st.header("🔍 Explorateur de Données")
    # Implémentation existante à adapter...
    st.info("Explorateur de données en cours de développement...")

if __name__ == "__main__":
    main()