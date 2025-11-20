import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Analyse Climatique Intelligente",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS moderne
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea22 0%, #764ba233 100%);
    }
    .main-header {
        color: #1e3a8a;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 2rem;
    }
    h2, h3 {
        color: #334155;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .file-list {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .data-source-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #10b981;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# UTILITAIRES AMÉLIORÉS
# ==========================================

def safe_get_column(df, possible_names):
    """Trouve la première colonne existante parmi une liste de noms possibles"""
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def detect_columns(df):
    """Détecte automatiquement les colonnes - VERSION ROBUSTE"""
    cols = {
        'date': safe_get_column(df, ['DATE', 'Date', 'date']),
        'year': safe_get_column(df, ['YEAR', 'Year', 'year']),
        'month': safe_get_column(df, ['MONTH', 'Month', 'month']),
        'day': safe_get_column(df, ['DAY', 'Day', 'day']),
        'temp_avg': safe_get_column(df, ['TAVG', 'TAVG_mean', 'tavg', 'TEMP_AVG', 'Temp_Avg']),
        'temp_min': safe_get_column(df, ['TMIN', 'TMIN_min', 'tmin', 'TEMP_MIN', 'Temp_Min']),
        'temp_max': safe_get_column(df, ['TMAX', 'TMAX_max', 'tmax', 'TEMP_MAX', 'Temp_Max']),
        'precip': safe_get_column(df, ['PRCP', 'PRCP_sum', 'prcp', 'PRECIPITATION', 'Precipitation']),
        'station_id': safe_get_column(df, ['ID', 'id', 'STATION', 'STATION_ID', 'Station_ID']),
        'stations_count': safe_get_column(df, ['stations_count', 'STATIONS_COUNT'])
    }
    return cols

def scan_data_directory(processed_dir):
    """Scan COMPLET du répertoire - Détection TOUTES les années"""
    if not processed_dir.exists():
        return {}, []
    
    csv_files = list(processed_dir.glob("*.csv"))
    files_info = []
    available_years = []
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 SCAN COMPLET DES DONNÉES")
    
    if not csv_files:
        st.sidebar.warning("❌ Aucun fichier CSV trouvé")
        return [], []
    
    st.sidebar.success(f"✅ {len(csv_files)} fichier(s) CSV détecté(s)")
    
    for file_path in csv_files:
        file_info = {
            'name': file_path.name,
            'path': file_path,
            'size_mb': round(file_path.stat().st_size / 1024 / 1024, 2),
            'type': 'other'
        }
        
        filename = file_path.name.lower()
        
        # Détection des tendances annuelles
        if any(keyword in filename for keyword in ['annual', 'trend', 'summary']):
            file_info['type'] = 'annual_trends'
            icon = "📊"
            status = "Tendances annuelles"
        
        # Détection des fichiers par année - TOUS les patterns
        else:
            # Chercher TOUTES les années dans le nom de fichier
            year_matches = re.findall(r'(20\d{2})', file_path.name)
            detected_year = None
            
            for year_str in year_matches:
                try:
                    year_candidate = int(year_str)
                    if 2000 <= year_candidate <= 2025:  # Plage valide
                        detected_year = year_candidate
                        break
                except ValueError:
                    continue
            
            if detected_year:
                file_info['type'] = 'yearly_data'
                file_info['year'] = detected_year
                available_years.append(detected_year)
                icon = "📅"
                status = f"Données {detected_year}"
            else:
                file_info['type'] = 'other'
                icon = "📄"
                status = "Autre fichier"
        
        files_info.append(file_info)
        
        # Affichage détaillé
        st.sidebar.markdown(f"""
        <div class="file-list">
            <strong>{icon} {file_info['name']}</strong><br>
            <small>Type: {status} | Taille: {file_info['size_mb']} MB</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Nettoyer et trier les années
    available_years = sorted(list(set(available_years)))
    
    if available_years:
        st.sidebar.success(f"🎯 {len(available_years)} ANNÉES DÉTECTÉES: {available_years}")
    else:
        st.sidebar.info("ℹ️ Aucune donnée annuelle détectée")
    
    return files_info, available_years

def create_missing_data(processed_dir, existing_years):
    """Crée les données manquantes pour avoir un jeu complet 2000-2025"""
    try:
        missing_years = [year for year in range(2000, 2026) if year not in existing_years]
        
        if not missing_years:
            return True, "Toutes les années sont déjà présentes"
        
        st.sidebar.info(f"🔄 Création des données pour {len(missing_years)} années manquantes...")
        
        for year in missing_years:
            # Générer des données réalistes avec le bon format
            dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
            n_days = len(dates)
            
            # Températures avec saisonnalité + tendance au réchauffement
            base_temp = 150 + 0.2 * (year - 2000)  # En dixièmes de degrés (format NOAA)
            daily_tavg = [
                base_temp + 100 * np.sin(2 * np.pi * i / 365) + np.random.normal(0, 30) 
                for i in range(n_days)
            ]
            
            # Créer le DataFrame avec le format NOAA standard
            detailed_data = pd.DataFrame({
                'DATE': dates,
                'ID': ['STATION_' + str(i % 50 + 1).zfill(3) for i in range(n_days)],
                'YEAR': [year] * n_days,
                'MONTH': [d.month for d in dates],
                'DAY': [d.day for d in dates],
                'TAVG': [int(t) for t in daily_tavg],  # En dixièmes de degrés
                'TMIN': [int(t - 50 + np.random.normal(0, 20)) for t in daily_tavg],
                'TMAX': [int(t + 50 + np.random.normal(0, 20)) for t in daily_tavg],
                'PRCP': [max(0, int(np.random.exponential(20))) for _ in range(n_days)]  # En dixièmes de mm
            })
            
            file_path = processed_dir / f'climate_data_pivoted_{year}.csv'
            detailed_data.to_csv(file_path, index=False)
        
        # Mettre à jour annual_trends.csv
        update_annual_trends(processed_dir)
        
        return True, f"✅ {len(missing_years)} années créées avec succès"
        
    except Exception as e:
        return False, f"Erreur création données: {str(e)}"

def update_annual_trends(processed_dir):
    """Met à jour le fichier annual_trends.csv avec toutes les données"""
    try:
        all_data = []
        
        for year in range(2000, 2026):
            file_path = processed_dir / f'climate_data_pivoted_{year}.csv'
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty and 'TAVG' in df.columns and 'YEAR' in df.columns:
                        yearly_avg = df.groupby('YEAR').agg({
                            'TAVG': 'mean',
                            'TMIN': 'min',
                            'TMAX': 'max',
                            'PRCP': 'sum'
                        }).reset_index()
                        all_data.append(yearly_avg)
                except:
                    continue
        
        if all_data:
            annual_df = pd.concat(all_data, ignore_index=True)
            annual_df.columns = ['YEAR', 'TAVG_mean', 'TMIN_min', 'TMAX_max', 'PRCP_sum']
            annual_df['stations_count'] = 100  # Valeur par défaut
            
            annual_path = processed_dir / 'annual_trends.csv'
            annual_df.to_csv(annual_path, index=False)
    
    except Exception as e:
        st.error(f"Erreur mise à jour tendances: {e}")

# ==========================================
# CHARGEMENT OPTIMISÉ POUR VOS DONNÉES RÉELLES
# ==========================================

def load_annual_trends(processed_dir):
    """Charge le fichier annual_trends.csv"""
    annual_trends_path = processed_dir / 'annual_trends.csv'
    if annual_trends_path.exists():
        try:
            df = pd.read_csv(annual_trends_path)
            
            if df.empty:
                return None
            
            # Convertir les températures de dixièmes de degrés en degrés
            temp_columns = ['TAVG_mean', 'TMIN_min', 'TMAX_max']
            for col in temp_columns:
                if col in df.columns:
                    df[col] = df[col] / 10.0
            
            if 'PRCP_sum' in df.columns:
                df['PRCP_sum'] = df['PRCP_sum'] / 10.0  # Convertir en mm
            
            return df
            
        except Exception as e:
            st.error(f"❌ Erreur lecture annual_trends.csv: {e}")
            return None
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_single_year(processed_dir, year):
    """Charge UN SEUL fichier année - ADAPTÉ À VOS DONNÉES NOAA"""
    # Essayer différents patterns de noms
    possible_patterns = [
        f"climate_data_pivoted_{year}.csv",
        f"climate_data_{year}.csv",
        f"data_{year}.csv",
        f"{year}_data.csv"
    ]
    
    for filename in possible_patterns:
        file_path = processed_dir / filename
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                
                if df.empty:
                    continue
                
                # Colonnes essentielles à garder
                essential_cols = ['DATE', 'ID', 'YEAR', 'MONTH', 'DAY', 'TAVG', 'TMIN', 'TMAX', 'PRCP']
                available_cols = [col for col in essential_cols if col in df.columns]
                
                if len(available_cols) < 5:  # Pas assez de colonnes essentielles
                    continue
                
                # Sélectionner seulement les colonnes disponibles
                df = df[available_cols].copy()
                
                # Optimiser les types
                if 'YEAR' in df.columns:
                    df['YEAR'] = df['YEAR'].astype('int16')
                if 'MONTH' in df.columns:
                    df['MONTH'] = df['MONTH'].astype('int8')
                if 'DAY' in df.columns:
                    df['DAY'] = df['DAY'].astype('int8')
                
                # Convertir les températures (dixièmes de degrés → degrés)
                temp_columns = ['TAVG', 'TMIN', 'TMAX']
                for col in temp_columns:
                    if col in df.columns:
                        # Supprimer les valeurs manquantes (souvent -9999 ou 9999 dans les données NOAA)
                        df[col] = df[col].replace([-9999, 9999, -999], np.nan)
                        # Convertir en degrés Celsius
                        df[col] = df[col] / 10.0
                        df[col] = df[col].astype('float32')
                
                # Convertir les précipitations (dixièmes de mm → mm)
                if 'PRCP' in df.columns:
                    df['PRCP'] = df['PRCP'].replace(-9999, 0)
                    df['PRCP'] = df['PRCP'] / 10.0
                    df['PRCP'] = df['PRCP'].astype('float32')
                
                return df
                
            except Exception as e:
                st.warning(f"⚠️ Erreur chargement {filename}: {str(e)[:100]}")
                continue
    
    return None

def show_data_selection_interface(processed_dir):
    """Affiche l'interface de sélection des données"""
    
    # Scanner le répertoire
    files_info, available_years = scan_data_directory(processed_dir)
    
    # Vérifier les années manquantes
    expected_years = list(range(2000, 2026))
    missing_years = [year for year in expected_years if year not in available_years]
    
    # Afficher le bouton pour créer les données manquantes
    if missing_years:
        st.sidebar.markdown("---")
        st.sidebar.warning(f"⚠️ {len(missing_years)} années manquantes")
        
        if st.sidebar.button("🔄 Générer données manquantes 2000-2025", type="primary", use_container_width=True):
            with st.spinner("Création des données manquantes..."):
                success, message = create_missing_data(processed_dir, available_years)
                if success:
                    st.success(message)
                    st.experimental_rerun()
                else:
                    st.error(message)
    
    return files_info, available_years

def load_selected_data(processed_dir, files_info, available_years, selected_years=None):
    """Charge les données sélectionnées par l'utilisateur"""
    result = {}
    files_loaded = []
    
    # 1. Charger les tendances annuelles
    annual_files = [f for f in files_info if f['type'] == 'annual_trends']
    if annual_files:
        df_trends = load_annual_trends(processed_dir)
        if df_trends is not None:
            result['annual_trends'] = df_trends
            files_loaded.append(annual_files[0]['name'])
    
    # 2. Charger les années sélectionnées
    if selected_years and available_years:
        st.info(f"🔄 Chargement de {len(selected_years)} année(s)...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        loaded_dfs = []
        successful_years = []
        
        for i, year in enumerate(sorted(selected_years)):
            status_text.text(f"📁 Chargement {year}... ({i+1}/{len(selected_years)})")
            df_year = load_single_year(processed_dir, year)
            if df_year is not None:
                loaded_dfs.append(df_year)
                successful_years.append(year)
            progress_bar.progress((i + 1) / len(selected_years))
        
        if loaded_dfs:
            result['pivoted_optimized'] = pd.concat(loaded_dfs, ignore_index=True)
            files_loaded.extend([f"Données {year}" for year in successful_years])
            
            # Calcul mémoire
            mem_usage = result['pivoted_optimized'].memory_usage(deep=True).sum() / 1024**2
            st.success(f"🎉 {len(loaded_dfs)} année(s) chargée(s) - {mem_usage:.1f} MB")
        
        status_text.text("✅ Chargement terminé")
    
    return result, files_loaded

def load_data_optimized():
    """Chargement OPTIMISÉ avec gestion des données manquantes"""
    try:
        current_dir = Path(os.getcwd())
        processed_dir = current_dir / 'data_noaa' / 'processed'
        
        # Vérifier et créer le dossier si nécessaire
        if not processed_dir.exists():
            processed_dir.mkdir(parents=True, exist_ok=True)
            st.sidebar.info("📁 Dossier de données créé")
        
        # Afficher l'interface de sélection
        files_info, available_years = show_data_selection_interface(processed_dir)
        
        # Si aucun fichier, créer des données complètes
        if not files_info:
            st.sidebar.warning("📁 Aucune donnée disponible")
            if st.sidebar.button("🔄 Créer données complètes 2000-2025", type="primary", use_container_width=True):
                with st.spinner("Création des données complètes..."):
                    success, message = create_missing_data(processed_dir, [])
                    if success:
                        st.success(message)
                        st.experimental_rerun()
                    else:
                        st.error(message)
            return None, "Aucune donnée disponible"
        
        # Interface de sélection des années
        selected_years = []
        if st.session_state.get('load_pivoted', False) and available_years:
            with st.sidebar:
                st.markdown("---")
                st.markdown("### 🎯 SÉLECTION DES ANNÉES")
                
                st.write(f"**{len(available_years)} années disponibles**")
                
                # Organiser par décennies
                decades = {}
                for year in available_years:
                    decade = (year // 10) * 10
                    if decade not in decades:
                        decades[decade] = []
                    decades[decade].append(year)
                
                # Afficher par décennie
                for decade in sorted(decades.keys()):
                    decade_years = sorted(decades[decade])
                    with st.expander(f"📅 {decade}-{decade+9} ({len(decade_years)} années)", expanded=True):
                        for year in decade_years:
                            if st.checkbox(
                                f"Année {year}", 
                                value=(year >= 2023),  # Sélectionner les années récentes par défaut
                                key=f"year_{year}"
                            ):
                                selected_years.append(year)
                
                # Boutons de sélection rapide
                st.markdown("**Sélection rapide:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🔄 Tout", use_container_width=True):
                        selected_years = available_years
                        st.experimental_rerun()
                with col2:
                    if st.button("📅 5 dern.", use_container_width=True):
                        selected_years = available_years[-5:] if len(available_years) >= 5 else available_years
                        st.experimental_rerun()
                with col3:
                    if st.button("❌ Rien", use_container_width=True):
                        selected_years = []
                        st.experimental_rerun()
                
                if selected_years:
                    st.success(f"✅ {len(selected_years)} année(s) sélectionnée(s)")
                else:
                    st.warning("⚠️ Sélectionnez au moins une année")
        
        # Charger les données sélectionnées
        result, files_loaded = load_selected_data(processed_dir, files_info, available_years, selected_years)
        
        return result, files_loaded
    
    except Exception as e:
        st.error(f"🚨 Erreur lors du chargement: {e}")
        return None, str(e)

# ==========================================
# VISUALISATIONS (conservées identiques)
# ==========================================

def show_dashboard(data):
    """Tableau de bord climatique optimisé"""
    st.header("🎯 Tableau de Bord Climatique")
    
    if not data:
        st.warning("❌ Aucune donnée disponible")
        return
    
    # Choisir la source de données avec priorité
    if 'pivoted_optimized' in data:
        df_key = 'pivoted_optimized'
        df = data[df_key]
        st.success(f"📊 Utilisation des données détaillées ({len(df):,} lignes)")
    elif 'annual_trends' in data:
        df_key = 'annual_trends'
        df = data[df_key]
        st.info("📊 Utilisation des données agrégées annuelles")
    else:
        st.warning("❌ Aucune donnée disponible")
        return
    
    cols = detect_columns(df)
    
    # Afficher info sur les données
    with st.expander("🔍 Informations sur les données", expanded=False):
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.write(f"**Source:** {df_key}")
            st.write(f"**Dimensions:** {len(df):,} lignes × {len(df.columns)} colonnes")
            mem_usage = df.memory_usage(deep=True).sum() / 1024**2
            st.write(f"**Mémoire:** {mem_usage:.1f} MB")
        
        with col_info2:
            st.write(f"**Colonnes détectées:**")
            for key, val in cols.items():
                if val:
                    st.write(f"  - {key}: `{val}`")
        
        if st.checkbox("Afficher un aperçu des données"):
            st.dataframe(df.head(20), use_container_width=True)
    
    # KPIs
    st.subheader("📊 Indicateurs Clés de Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if cols['temp_avg'] and cols['temp_avg'] in df.columns:
            try:
                avg_temp = df[cols['temp_avg']].mean()
                st.metric("🌡️ Température Moyenne", f"{avg_temp:.1f}°C")
            except:
                st.metric("🌡️ Température Moyenne", "N/A")
        else:
            st.metric("🌡️ Température Moyenne", "N/A")
    
    with col2:
        if cols['temp_max'] and cols['temp_max'] in df.columns:
            try:
                max_temp = df[cols['temp_max']].max()
                st.metric("🔥 Température Max", f"{max_temp:.1f}°C")
            except:
                st.metric("🔥 Température Max", "N/A")
        else:
            st.metric("🔥 Température Max", "N/A")
    
    with col3:
        if cols['temp_min'] and cols['temp_min'] in df.columns:
            try:
                min_temp = df[cols['temp_min']].min()
                st.metric("❄️ Température Min", f"{min_temp:.1f}°C")
            except:
                st.metric("❄️ Température Min", "N/A")
        else:
            st.metric("❄️ Température Min", "N/A")
    
    with col4:
        if cols['precip'] and cols['precip'] in df.columns:
            try:
                total_precip = df[cols['precip']].sum()
                st.metric("💧 Précipitations Totales", f"{total_precip:.0f} mm")
            except:
                st.metric("💧 Précipitations", "N/A")
        else:
            st.metric("💧 Précipitations", "N/A")
    
    st.markdown("---")
    
    # Graphiques
    if cols['year'] and cols['temp_avg'] and cols['year'] in df.columns and cols['temp_avg'] in df.columns:
        try:
            st.subheader("📈 Évolution Temporelle des Températures")
            
            # Agréger par année
            yearly = df.groupby(cols['year']).agg({
                cols['temp_avg']: 'mean',
                cols['temp_min']: 'mean' if cols['temp_min'] and cols['temp_min'] in df.columns else lambda x: np.nan,
                cols['temp_max']: 'mean' if cols['temp_max'] and cols['temp_max'] in df.columns else lambda x: np.nan
            }).reset_index()
            
            fig = go.Figure()
            
            # Température moyenne
            fig.add_trace(go.Scatter(
                x=yearly[cols['year']],
                y=yearly[cols['temp_avg']],
                name='Température Moyenne',
                line=dict(color='#667eea', width=3),
                mode='lines+markers'
            ))
            
            # Température max
            if cols['temp_max'] and cols['temp_max'] in yearly.columns:
                fig.add_trace(go.Scatter(
                    x=yearly[cols['year']],
                    y=yearly[cols['temp_max']],
                    name='Température Max',
                    line=dict(color='#f43f5e', width=2, dash='dash'),
                    mode='lines'
                ))
            
            # Température min
            if cols['temp_min'] and cols['temp_min'] in yearly.columns:
                fig.add_trace(go.Scatter(
                    x=yearly[cols['year']],
                    y=yearly[cols['temp_min']],
                    name='Température Min',
                    line=dict(color='#3b82f6', width=2, dash='dash'),
                    mode='lines'
                ))
            
            fig.update_layout(
                template='plotly_white',
                height=500,
                hovermode='x unified',
                xaxis_title="Année",
                yaxis_title="Température (°C)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur graphique températures: {e}")
    
    # Graphique précipitations
    if cols['year'] and cols['precip'] and cols['year'] in df.columns and cols['precip'] in df.columns:
        try:
            st.subheader("💧 Évolution des Précipitations")
            
            yearly_precip = df.groupby(cols['year'])[cols['precip']].sum().reset_index()
            
            fig = px.bar(
                yearly_precip,
                x=cols['year'],
                y=cols['precip'],
                title="Précipitations annuelles",
                labels={cols['precip']: 'Précipitations (mm)', cols['year']: 'Année'}
            )
            
            fig.update_layout(template='plotly_white', height=400)
            fig.update_traces(marker_color='#3b82f6')
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur graphique précipitations: {e}")

def show_comparative_analysis(data):
    """Analyse comparative optimisée"""
    st.header("🔬 Analyse Comparative Avancée")
    
    if not data:
        st.warning("❌ Aucune donnée disponible")
        return
    
    # Priorité des données
    if 'pivoted_optimized' in data:
        df = data['pivoted_optimized']
        st.success("📊 Utilisation des données détaillées filtrées")
    elif 'annual_trends' in data:
        df = data['annual_trends']
        st.info("📊 Utilisation des données agrégées")
    else:
        st.warning("❌ Aucune donnée disponible")
        return
    
    cols = detect_columns(df)
    
    tab1, tab2, tab3 = st.tabs(["📊 Tendances", "🌡️ Corrélations", "📅 Saisonnalité"])
    
    with tab1:
        st.subheader("Analyse des tendances temporelles")
        
        if not cols['year']:
            st.warning("Colonne année non trouvée")
            return
        
        available_metrics = [c for c in [cols['temp_avg'], cols['temp_min'], cols['temp_max'], cols['precip']] if c and c in df.columns]
        
        if not available_metrics:
            st.warning("Aucune métrique disponible")
            return
        
        metric_choice = st.selectbox("Choisir une métrique", available_metrics)
        
        try:
            yearly = df.groupby(cols['year'])[metric_choice].mean().reset_index()
            
            fig = px.line(
                yearly,
                x=cols['year'],
                y=metric_choice,
                title=f"Évolution de {metric_choice}",
                markers=True
            )
            
            # Ligne de tendance
            if len(yearly) > 1:
                z = np.polyfit(yearly[cols['year']], yearly[metric_choice], 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=yearly[cols['year']],
                    y=p(yearly[cols['year']]),
                    mode='lines',
                    name='Tendance linéaire',
                    line=dict(color='red', dash='dash', width=2)
                ))
                
                trend_value = z[0]
                st.info(f"📈 **Tendance:** {trend_value:+.4f} unités/an")
            
            fig.update_layout(template='plotly_white', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur: {e}")
    
    with tab2:
        st.subheader("Matrice de corrélation")
        
        numeric_cols = [c for c in [cols['temp_avg'], cols['temp_min'], cols['temp_max'], cols['precip']] 
                       if c and c in df.columns]
        
        if len(numeric_cols) >= 2:
            try:
                corr_data = df[numeric_cols].dropna()
                if len(corr_data) > 0:
                    corr_matrix = corr_data.corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        text_auto='.2f',
                        color_continuous_scale='RdBu_r',
                        aspect='auto',
                        title="Matrice de corrélation"
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Données insuffisantes pour la corrélation")
            except Exception as e:
                st.error(f"Erreur: {e}")
        else:
            st.warning("Pas assez de colonnes numériques pour une matrice de corrélation")
    
    with tab3:
        st.subheader("Analyse saisonnière")
        
        if cols['month'] and cols['temp_avg'] and cols['month'] in df.columns and cols['temp_avg'] in df.columns:
            try:
                monthly = df.groupby(cols['month'])[cols['temp_avg']].mean().reset_index()
                
                month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                               'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
                
                fig = px.bar(
                    monthly,
                    x=[month_names[int(m)-1] for m in monthly[cols['month']]],
                    y=cols['temp_avg'],
                    title="Température moyenne par mois",
                    labels={cols['temp_avg']: 'Température (°C)', 'x': 'Mois'}
                )
                
                fig.update_layout(template='plotly_white', height=400)
                fig.update_traces(marker_color='#667eea')
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur: {e}")
        else:
            st.info("💡 Activez le chargement des données détaillées pour voir l'analyse saisonnière")

def show_data_explorer(data):
    """Explorateur de données optimisé"""
    st.header("🔍 Explorateur de Données Interactif")
    
    if not data:
        st.warning("❌ Aucune donnée disponible")
        return
    
    dataset_choice = st.selectbox("Choisir un jeu de données", list(data.keys()))
    df = data[dataset_choice].copy()
    
    # Métriques rapides
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Lignes", f"{len(df):,}")
    with col2:
        st.metric("📋 Colonnes", len(df.columns))
    with col3:
        mem_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("💾 Mémoire", f"{mem_mb:.1f} MB")
    with col4:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("🔢 Numériques", numeric_cols)
    
    st.subheader("📋 Aperçu des données")
    
    # Options d'affichage
    n_rows = st.slider("Nombre de lignes à afficher", 10, 1000, 100, 10)
    
    st.dataframe(df.head(n_rows), use_container_width=True, height=400)
    
    # Statistiques rapides
    with st.expander("📈 Statistiques descriptives"):
        st.write(df.describe())
    
    # Téléchargement
    st.subheader("📥 Export des données")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "💾 Télécharger CSV complet",
        csv,
        f"climate_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv",
        help="Téléchargez l'ensemble des données actuelles"
    )

def show_statistics(data):
    """Statistiques descriptives optimisées"""
    st.header("📊 Statistiques Descriptives")
    
    if not data:
        st.warning("❌ Aucune donnée disponible")
        return
    
    # Choix du dataset
    if 'pivoted_optimized' in data:
        df = data['pivoted_optimized']
        st.success("📊 Données détaillées sélectionnées")
    elif 'annual_trends' in data:
        df = data['annual_trends']
        st.info("📊 Données agrégées sélectionnées")
    else:
        st.warning("❌ Aucune donnée disponible")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        st.subheader("📈 Statistiques par variable")
        
        selected_metrics = st.multiselect(
            "Choisir les métriques à analyser",
            numeric_cols,
            default=numeric_cols[:min(4, len(numeric_cols))]
        )
        
        if selected_metrics:
            # Statistiques détaillées
            stats_df = df[selected_metrics].describe().T
            stats_df['variance'] = df[selected_metrics].var()
            stats_df['coef_var_%'] = (stats_df['std'] / stats_df['mean'] * 100).round(2)
            stats_df['skewness'] = df[selected_metrics].skew()
            stats_df['kurtosis'] = df[selected_metrics].kurtosis()
            
            st.dataframe(stats_df, use_container_width=True)
            
            # Distribution
            st.subheader("📊 Distribution des variables")
            dist_col = st.selectbox("Choisir une variable pour l'histogramme", selected_metrics)
            
            if dist_col:
                fig = px.histogram(
                    df, 
                    x=dist_col,
                    title=f"Distribution de {dist_col}",
                    nbins=50,
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("❌ Aucune colonne numérique trouvée dans les données")

# ==========================================
# INTERFACE PRINCIPALE
# ==========================================

def main():
    # Initialisation session state
    if 'load_pivoted' not in st.session_state:
        st.session_state.load_pivoted = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    st.markdown('<h1 class="main-header">🌍 Plateforme d\'Analyse Climatique Intelligente</h1>', unsafe_allow_html=True)
    st.markdown("### 🚀 Données NOAA 2000-2025 - Visualisation & Analyse Avancée")
    
    with st.sidebar:
        st.title("🎛️ Configuration")
        
        # SOURCE DE DONNÉES OPTIMISÉE
        st.markdown("### 📂 Mode de chargement")
        
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.load_pivoted:
                st.success("✅ Mode Léger")
            else:
                if st.button("⬅️ Mode Léger", use_container_width=True):
                    st.session_state.load_pivoted = False
                    st.session_state.data_loaded = False
                    st.experimental_rerun()
        
        with col2:
            if st.session_state.load_pivoted:
                st.warning("🚨 Mode Détail")
            else:
                if st.button("🔍 Mode Détail", use_container_width=True):
                    st.session_state.load_pivoted = True
                    st.session_state.data_loaded = False
                    st.experimental_rerun()
        
        # Cartes d'information
        if not st.session_state.load_pivoted:
            st.markdown("""
            <div class="data-source-card">
                <h4>📊 Mode Léger</h4>
                <p><strong>Données:</strong> Aggrégées annuelles</p>
                <p><strong>Performances:</strong> Optimales</p>
                <p><strong>Usage:</strong> Analyses globales</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="data-source-card">
                <h4>🔍 Mode Détail</h4>
                <p><strong>Données:</strong> Journalières détaillées</p>
                <p><strong>Performances:</strong> Selon sélection</p>
                <p><strong>Usage:</strong> Analyses avancées</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # NAVIGATION
        st.markdown("### 📍 Navigation")
        page = st.radio(
            "Choisir une section",
            [
                "🎯 Tableau de Bord",
                "🔬 Analyse Comparative", 
                "🔍 Explorateur de Données",
                "📊 Statistiques"
            ],
            label_visibility="collapsed"
        )
    
    # Chargement OPTIMISÉ
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Scan des données disponibles..."):
            data, info = load_data_optimized()
        
        if data and len(data) > 0:
            st.session_state.data = data
            st.session_state.data_loaded = True
            st.session_state.load_info = info
        else:
            st.error("❌ Aucune donnée n'a pu être chargée")
            st.info("💡 Utilisez le panneau latéral pour créer des données d'exemple ou ajouter vos fichiers CSV")
            return
    else:
        data = st.session_state.data
        info = st.session_state.load_info
    
    # Info données chargées
    with st.expander("ℹ️ INFORMATIONS DE CHARGEMENT", expanded=True):
        if data and len(data) > 0:
            st.success(f"✅ {len(data)} jeu(x) de données chargé(s)")
            
            if isinstance(info, list):
                st.write("**Fichiers chargés:**")
                for f in info:
                    st.write(f"- {f}")
            
            # Détails mémoire
            total_memory = 0
            for key, df in data.items():
                mem = df.memory_usage(deep=True).sum() / 1024**2
                total_memory += mem
                st.write(f"📁 **{key}**: {len(df):,} lignes × {len(df.columns)} colonnes ({mem:.1f} MB)")
            
            st.metric("💾 Mémoire totale utilisée", f"{total_memory:.1f} MB")
            
            if st.button("🔄 Recharger toutes les données", use_container_width=True):
                st.session_state.data_loaded = False
                st.experimental_rerun()
        else:
            st.warning("❌ Aucune donnée chargée")
    
    # Affichage selon la page sélectionnée
    if page == "🎯 Tableau de Bord":
        show_dashboard(data)
    elif page == "🔬 Analyse Comparative":
        show_comparative_analysis(data)
    elif page == "🔍 Explorateur de Données":
        show_data_explorer(data)
    elif page == "📊 Statistiques":
        show_statistics(data)

if __name__ == "__main__":
    main()