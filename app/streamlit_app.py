import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import joblib
import requests
import io
from typing import Dict, List, Optional
import urllib.parse
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

# Configuration NOAA
NOAA_CONFIG = {
    'base_url': 'https://www.ncei.noaa.gov/access/services/data/v1',
    'dataset': 'global-summary-of-the-day',
    'data_types': ['TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 'MXSPD', 'GUST', 'MAX', 'MIN', 'PRCP', 'SNDP', 'FRSHTT'],
    'stations': {
        'FR': ['071490-99999', '071560-99999', '071570-99999', '071580-99999'],  # Stations françaises exemple
        'US': ['725030-14732', '724080-13722'],  # Stations US exemple
        'default': ['725030-14732']  # New York Central Park par défaut
    }
}

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
    .noaa-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        text-align: center;
    }
    .status-success {
        background: #c6f6d5;
        color: #22543d;
        border-left: 4px solid #38a169;
    }
    .status-warning {
        background: #feebc8;
        color: #744210;
        border-left: 4px solid #ed8936;
    }
    .status-error {
        background: #fed7d7;
        color: #742a2a;
        border-left: 4px solid #e53e3e;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# FONCTIONS DE CHARGEMENT DES DONNÉES NOAA EN LIGNE
# ==========================================

class NOAADataLoader:
    """Chargeur de données NOAA en ligne"""
    
    def __init__(self):
        self.base_url = NOAA_CONFIG['base_url']
        self.dataset = NOAA_CONFIG['dataset']
        
    def build_noaa_url(self, stations: List[str], start_date: str, end_date: str, 
                      data_types: List[str] = None) -> str:
        """Construit l'URL pour l'API NOAA"""
        if data_types is None:
            data_types = ['TEMP', 'MAX', 'MIN', 'PRCP']
            
        stations_param = ','.join(stations)
        data_types_param = ','.join(data_types)
        
        params = {
            'dataset': self.dataset,
            'stations': stations_param,
            'startDate': start_date,
            'endDate': end_date,
            'dataTypes': data_types_param,
            'format': 'csv',
            'units': 'metric'
        }
        
        query_string = urllib.parse.urlencode(params, doseq=True)
        return f"{self.base_url}?{query_string}"
    
    def fetch_noaa_data(self, stations: List[str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Récupère les données depuis l'API NOAA"""
        try:
            url = self.build_noaa_url(stations, start_date, end_date)
            st.sidebar.info(f"🌐 Chargement des données depuis NOAA...")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Vérifier si la réponse contient des données
            if len(response.content) < 100:  # Réponse trop courte = probablement pas de données
                return None
                
            df = pd.read_csv(io.StringIO(response.text))
            
            if df.empty:
                return None
                
            # Nettoyage des données
            df = self.clean_noaa_data(df)
            return df
            
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"❌ Erreur réseau: {e}")
            return None
        except pd.errors.EmptyDataError:
            st.sidebar.warning("⚠️ Aucune donnée disponible pour cette période")
            return None
        except Exception as e:
            st.sidebar.error(f"❌ Erreur inattendue: {e}")
            return None
    
    def clean_noaa_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie et transforme les données NOAA"""
        # Copie pour éviter les warnings
        df_clean = df.copy()
        
        # Colonnes standard NOAA
        column_mapping = {
            'STATION': 'ID',
            'DATE': 'DATE',
            'LATITUDE': 'LATITUDE',
            'LONGITUDE': 'LONGITUDE',
            'ELEVATION': 'ELEVATION',
            'NAME': 'NAME',
            'TEMP': 'TAVG',
            'MAX': 'TMAX', 
            'MIN': 'TMIN',
            'PRCP': 'PRCP'
        }
        
        # Renommer les colonnes
        df_clean = df_clean.rename(columns={k: v for k, v in column_mapping.items() if k in df_clean.columns})
        
        # Conversion des dates
        if 'DATE' in df_clean.columns:
            df_clean['DATE'] = pd.to_datetime(df_clean['DATE'])
            df_clean['YEAR'] = df_clean['DATE'].dt.year
            df_clean['MONTH'] = df_clean['DATE'].dt.month
            df_clean['DAY'] = df_clean['DATE'].dt.day
        
        # Conversion des températures (dixièmes de degrés → degrés)
        temp_columns = ['TAVG', 'TMAX', 'TMIN']
        for col in temp_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col] / 10.0
                # Supprimer les valeurs extrêmes
                df_clean[col] = df_clean[col].replace([-999.9, 999.9], np.nan)
        
        # Conversion des précipitations (dixièmes de mm → mm)
        if 'PRCP' in df_clean.columns:
            df_clean['PRCP'] = pd.to_numeric(df_clean['PRCP'], errors='coerce')
            df_clean['PRCP'] = df_clean['PRCP'] / 10.0
            df_clean['PRCP'] = df_clean['PRCP'].replace(-999.9, 0)
        
        # Supprimer les lignes avec trop de valeurs manquantes
        required_cols = [col for col in ['TAVG', 'TMAX', 'TMIN', 'PRCP'] if col in df_clean.columns]
        if required_cols:
            df_clean = df_clean.dropna(subset=required_cols, how='all')
        
        return df_clean.reset_index(drop=True)

# ==========================================
# FONCTIONS UTILITAIRES AMÉLIORÉES
# ==========================================

def get_default_dates():
    """Retourne les dates par défaut pour le chargement"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365 * 5)  # 5 ans de données
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def show_noaa_status():
    """Affiche le statut de connexion NOAA"""
    st.sidebar.markdown("### 🌐 Statut NOAA")
    
    # Test de connexion
    try:
        test_url = NOAA_CONFIG['base_url']
        response = requests.get(test_url, timeout=10)
        if response.status_code == 200:
            st.sidebar.markdown('<div class="noaa-status status-success">✅ Connecté à NOAA</div>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<div class="noaa-status status-warning">⚠️ Service NOAA limité</div>', unsafe_allow_html=True)
    except:
        st.sidebar.markdown('<div class="noaa-status status-error">❌ Hors ligne - Mode démo</div>', unsafe_allow_html=True)

def load_data_optimized():
    """Chargement principal des données depuis NOAA en ligne"""
    try:
        # Initialisation du chargeur
        loader = NOAADataLoader()
        
        # Interface de configuration dans la sidebar
        st.sidebar.markdown("### 📡 Configuration NOAA")
        
        # Sélection des stations
        country = st.sidebar.selectbox(
            "Pays des stations",
            list(NOAA_CONFIG['stations'].keys()),
            index=0
        )
        
        selected_stations = st.sidebar.multiselect(
            "Stations météo",
            NOAA_CONFIG['stations'][country],
            default=NOAA_CONFIG['stations'][country][:2]  # 2 premières stations par défaut
        )
        
        if not selected_stations:
            selected_stations = NOAA_CONFIG['stations']['default']
        
        # Sélection de la période
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Date de début",
                datetime.now() - timedelta(days=365 * 2),  # 2 ans par défaut
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "Date de fin",
                datetime.now(),
                max_value=datetime.now()
            )
        
        # Conversion en string
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Bouton de chargement
        if st.sidebar.button("📥 Charger les données NOAA", type="primary", use_container_width=True):
            with st.spinner(f"Chargement des données du {start_str} au {end_str}..."):
                df_noaa = loader.fetch_noaa_data(selected_stations, start_str, end_str)
                
                if df_noaa is not None and not df_noaa.empty:
                    result = {
                        'noaa_live_data': df_noaa,
                        'stations': selected_stations,
                        'period': f"{start_str} to {end_str}"
                    }
                    
                    # Création de données agrégées pour l'analyse
                    result['annual_trends'] = create_annual_trends(df_noaa)
                    result['monthly_data'] = create_monthly_aggregates(df_noaa)
                    
                    st.sidebar.success(f"✅ {len(df_noaa)} enregistrements chargés")
                    return result, f"Données NOAA {len(selected_stations)} stations"
                else:
                    st.sidebar.error("❌ Aucune donnée récupérée")
                    return {}, "Échec chargement"
        
        # Retourner les données de démo si pas de chargement
        return load_demo_data(), "Mode démo - Configurez et chargez les données"
        
    except Exception as e:
        st.error(f"🚨 Erreur chargement données: {e}")
        return load_demo_data(), f"Mode démo - Erreur: {str(e)}"

def create_annual_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des tendances annuelles à partir des données quotidiennes"""
    if df.empty or 'YEAR' not in df.columns:
        return pd.DataFrame()
    
    trends = df.groupby('YEAR').agg({
        'TAVG': ['mean', 'std'],
        'TMAX': 'max',
        'TMIN': 'min', 
        'PRCP': 'sum'
    }).round(2)
    
    # Aplatir les colonnes
    trends.columns = [f'{col[0]}_{col[1]}'.upper() for col in trends.columns]
    trends = trends.reset_index()
    
    return trends

def create_monthly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des agrégats mensuels"""
    if df.empty or 'MONTH' not in df.columns:
        return pd.DataFrame()
    
    monthly = df.groupby(['YEAR', 'MONTH']).agg({
        'TAVG': 'mean',
        'TMAX': 'max',
        'TMIN': 'min',
        'PRCP': 'sum'
    }).reset_index()
    
    return monthly

def load_demo_data():
    """Charge des données de démo si NOAA n'est pas disponible"""
    # Génération de données de démo réalistes
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    
    demo_data = pd.DataFrame({
        'DATE': dates,
        'YEAR': dates.year,
        'MONTH': dates.month,
        'DAY': dates.day,
        'ID': 'DEMO_STATION',
        'TAVG': np.random.normal(15, 5, len(dates)),  # Température moyenne ~15°C
        'TMAX': np.random.normal(20, 6, len(dates)),  # Température max ~20°C
        'TMIN': np.random.normal(10, 4, len(dates)),  # Température min ~10°C
        'PRCP': np.random.gamma(2, 2, len(dates)),    # Précipitations gamma distribuées
        'LATITUDE': 48.8566,
        'LONGITUDE': 2.3522,
        'NAME': 'Paris Demo Station'
    })
    
    # Ajouter de la saisonnalité
    demo_data['TAVG'] += 10 * np.sin(2 * np.pi * (demo_data['MONTH'] - 1) / 11)
    demo_data['TMAX'] += 12 * np.sin(2 * np.pi * (demo_data['MONTH'] - 1) / 11)
    demo_data['TMIN'] += 8 * np.sin(2 * np.pi * (demo_data['MONTH'] - 1) / 11)
    
    # Créer les agrégats
    annual_trends = create_annual_trends(demo_data)
    monthly_data = create_monthly_aggregates(demo_data)
    
    return {
        'noaa_live_data': demo_data,
        'annual_trends': annual_trends,
        'monthly_data': monthly_data,
        'stations': ['DEMO_STATION'],
        'period': '2020-2024 (Démo)'
    }

# ==========================================
# INTERFACES SPÉCIFIQUES AGRICULTURE (MAINTENUES)
# ==========================================

def show_project_overview():
    """Vue d'ensemble du projet Climat Impact Agricole"""
    st.markdown("""
    <div class="main-header">
        <h1>🌾 Climat Impact Agricole</h1>
        <h3>Analyse Intelligente des Données Climatiques NOAA en Temps Réel</h3>
        <p>Projet CRISP-DM - Master Data Science</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Nouvelle section pour les données NOAA en direct
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📡 Données NOAA en Temps Réel
        
        **Nouveauté :** Connexion directe à l'API NOAA pour des données actualisées
        - 📊 Données météo mondiales en direct
        - 🌡️ Températures, précipitations, humidité
        - 📍 Multiples stations météorologiques
        - ⏰ Historique jusqu'à présent
        
        **Fonctionnalités :**
        - Chargement personnalisé par période
        - Sélection de stations par pays
        - Analyse comparative multi-stations
        - Données nettoyées et standardisées
        """)
    
    with col2:
        st.markdown("### 🎯 Métriques Clés")
        st.metric("🌐 Source données", "NOAA Global")
        st.metric("📡 Stations disponibles", "50 000+")
        st.metric("📊 Données temps réel", "Oui")
        st.metric("🕐 Mise à jour", "Quotidienne")
    
    # Architecture CRISP-DM mise à jour
    st.markdown("### 🔧 Architecture CRISP-DM Améliorée")
    
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
            <p>• API NOAA en direct</p>
            <p>• Nettoyage automatique</p>
            <p>• Feature engineering</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="crispdm-phase">
            <h4>📁 2. Compréhension Données</h4>
            <p>• Exploration NOAA API</p>
            <p>• Qualité données temps réel</p>
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
            <p>• Données temps réel NOAA</p>
            <p>• Rapports automatiques</p>
        </div>
        """, unsafe_allow_html=True)

def show_noaa_dashboard(data):
    """Nouveau tableau de bord spécifique NOAA"""
    st.header("📡 Tableau de Bord NOAA en Temps Réel")
    
    if not data or 'noaa_live_data' not in data:
        st.warning("❌ Chargez d'abord les données NOAA")
        return
    
    df = data['noaa_live_data']
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latest_date = df['DATE'].max() if 'DATE' in df.columns else 'N/A'
        st.metric("📅 Dernière mise à jour", str(latest_date)[:10])
    
    with col2:
        stations_count = df['ID'].nunique() if 'ID' in df.columns else 0
        st.metric("📍 Stations", stations_count)
    
    with col3:
        avg_temp = df['TAVG'].mean() if 'TAVG' in df.columns else 0
        st.metric("🌡️ Temp. moyenne", f"{avg_temp:.1f}°C")
    
    with col4:
        total_precip = df['PRCP'].sum() if 'PRCP' in df.columns else 0
        st.metric("💧 Précipitations totales", f"{total_precip:.0f} mm")
    
    # Cartes des stations
    if all(col in df.columns for col in ['LATITUDE', 'LONGITUDE']):
        st.subheader("🗺️ Carte des Stations Météo")
        
        station_summary = df.groupby(['ID', 'LATITUDE', 'LONGITUDE', 'NAME']).agg({
            'TAVG': 'mean',
            'PRCP': 'sum'
        }).reset_index()
        
        fig_map = px.scatter_mapbox(
            station_summary,
            lat='LATITUDE',
            lon='LONGITUDE',
            hover_name='NAME',
            hover_data={'TAVG': ':.1f', 'PRCP': ':.0f', 'LATITUDE': ':.2f', 'LONGITUDE': ':.2f'},
            size='TAVG',
            color='PRCP',
            color_continuous_scale=px.colors.sequential.Viridis,
            zoom=3,
            height=400
        )
        fig_map.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig_map, use_container_width=True)
    
    # Tendances temporelles
    st.subheader("📈 Tendances Climatiques")
    
    if 'DATE' in df.columns and 'TAVG' in df.columns:
        # Températures dans le temps
        fig_temp = px.line(
            df, x='DATE', y=['TAVG', 'TMAX', 'TMIN'],
            title='Évolution des Températures',
            labels={'value': 'Température (°C)', 'variable': 'Type'}
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    
    # Analyse comparative des stations
    if 'ID' in df.columns and df['ID'].nunique() > 1:
        st.subheader("🏁 Comparaison des Stations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Températures par station
            station_temp = df.groupby('ID')['TAVG'].mean().sort_values(ascending=False)
            fig_station_temp = px.bar(
                station_temp,
                title='Température Moyenne par Station',
                labels={'value': 'Température (°C)', 'ID': 'Station'}
            )
            st.plotly_chart(fig_station_temp, use_container_width=True)
        
        with col2:
            # Précipitations par station
            station_precip = df.groupby('ID')['PRCP'].sum().sort_values(ascending=False)
            fig_station_precip = px.bar(
                station_precip,
                title='Précipitations Totales par Station',
                labels={'value': 'Précipitations (mm)', 'ID': 'Station'}
            )
            st.plotly_chart(fig_station_precip, use_container_width=True)

# ==========================================
# INTERFACE PRINCIPALE AMÉLIORÉE
# ==========================================

def main():
    # Initialisation session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Sidebar améliorée avec statut NOAA
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center;'>
            <h1>🌾 Climat Impact Agricole</h1>
            <p><em>Données NOAA en Temps Réel</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Statut NOAA
        show_noaa_status()
        
        st.markdown("### 🎛️ Navigation")
        page = st.radio(
            "Sélectionner une section",
            [
                "🏠 Vue d'ensemble",
                "📡 Dashboard NOAA", 
                "📈 Tableau de Bord",
                "🧠 Intelligence Artificielle",
                "🌱 Insights Agricoles",
                "📊 Qualité Données"
            ]
        )
        
        st.markdown("---")
        st.markdown("### 📂 Chargement Données")
        st.info("Configurez et chargez les données NOAA ci-dessous ⬇️")
    
    # Chargement des données
    if not st.session_state.data_loaded:
        data, info = load_data_optimized()
        
        if data:
            st.session_state.data = data
            st.session_state.data_loaded = True
            st.session_state.load_info = info
            st.sidebar.success(f"✅ {info}")
        else:
            st.error("❌ Échec du chargement des données")
            data = None
    else:
        data = st.session_state.data
    
    # Navigation améliorée
    if page == "🏠 Vue d'ensemble":
        show_project_overview()
    
    elif page == "📡 Dashboard NOAA":
        show_noaa_dashboard(data)
    
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Climat Impact Agricole</strong> - Données NOAA en temps réel</p>
        <p>🌐 <a href='https://www.noaa.gov/' target='_blank'>Source: NOAA Global Summary of the Day</a> | 
        📚 <a href='https://github.com/Adnane-dev/Climat_imapct_agricole' target='_blank'>GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FONCTIONS EXISTANTES (À ADAPTER)
# ==========================================

@st.cache_resource
def load_ml_models():
    """Charge les modèles ML depuis votre structure Git"""
    models = {}
    try:
        models_dir = project_root / '04_modeling' / 'models_saved'
        
        if not models_dir.exists():
            st.sidebar.warning("📁 Dossier des modèles non trouvé. Exécutez d'abord l'entraînement.")
            return {}
        
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

def show_dashboard(data):
    """Tableau de bord climatique existant"""
    st.header("📈 Tableau de Bord Climatique")
    # Votre implémentation existante...
    st.info("Tableau de bord en cours de développement...")

def show_ml_interface(data):
    """Interface Machine Learning"""
    st.header("🧠 Intelligence Artificielle Agricole")
    # Votre implémentation existante...
    st.info("Interface ML en cours de développement...")

def show_agricultural_insights(data):
    """Insights agricoles"""
    st.header("🌱 Insights Agricoles")
    # Votre implémentation existante...
    st.info("Insights agricoles en cours de développement...")

def show_data_quality_report(data):
    """Rapport qualité données"""
    st.header("📊 Qualité des Données")
    # Votre implémentation existante...
    st.info("Rapport qualité en cours de développement...")

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

if __name__ == "__main__":
    main()
