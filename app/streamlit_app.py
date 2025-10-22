import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
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
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
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
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# UTILITAIRES
# ==========================================

def safe_get_column(df, possible_names):
    """Trouve la première colonne existante parmi une liste de noms possibles"""
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def detect_columns(df):
    """Détecte automatiquement les colonnes - ADAPTÉ À VOS DONNÉES"""
    cols = {
        'date': safe_get_column(df, ['DATE', 'Date', 'date']),
        'year': safe_get_column(df, ['YEAR', 'Year', 'year']),
        'month': safe_get_column(df, ['MONTH', 'Month', 'month']),
        'day': safe_get_column(df, ['DAY', 'Day', 'day']),
        'temp_avg': safe_get_column(df, ['TAVG', 'TAVG_mean', 'tavg']),
        'temp_min': safe_get_column(df, ['TMIN', 'TMIN_min', 'tmin']),
        'temp_max': safe_get_column(df, ['TMAX', 'TMAX_max', 'tmax']),
        'precip': safe_get_column(df, ['PRCP', 'PRCP_sum', 'prcp']),
        'station_id': safe_get_column(df, ['ID', 'id', 'STATION']),
        'stations_count': safe_get_column(df, ['stations_count', 'STATIONS_COUNT'])
    }
    return cols

# ==========================================
# CHARGEMENT OPTIMISÉ
# ==========================================

def load_annual_trends(processed_dir):
    """Charge le fichier annual_trends.csv"""
    annual_trends_path = processed_dir / 'annual_trends.csv'
    if annual_trends_path.exists():
        try:
            df = pd.read_csv(annual_trends_path)
            # Convertir les températures de dixièmes de degrés en degrés
            if 'TAVG_mean' in df.columns:
                df['TAVG_mean'] = df['TAVG_mean'] / 10.0
            if 'TMIN_min' in df.columns:
                df['TMIN_min'] = df['TMIN_min'].replace(-999, np.nan) / 10.0
            if 'TMAX_max' in df.columns:
                df['TMAX_max'] = df['TMAX_max'] / 10.0
            if 'PRCP_sum' in df.columns:
                df['PRCP_sum'] = df['PRCP_sum'] / 10.0
            return df
        except Exception as e:
            st.error(f"Erreur annual_trends: {e}")
            return None
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_single_year(processed_dir, year):
    """Charge UN SEUL fichier année de façon optimisée"""
    file_path = processed_dir / f"climate_data_pivoted_{year}.csv"
    if file_path.exists():
        try:
            df = pd.read_csv(
                file_path,
                usecols=['DATE', 'ID', 'YEAR', 'MONTH', 'DAY', 'TAVG', 'TMIN', 'TMAX', 'PRCP'],
                dtype={
                    'YEAR': 'int16',
                    'MONTH': 'int8',
                    'DAY': 'int8',
                    'TAVG': 'float32',
                    'TMIN': 'float32',
                    'TMAX': 'float32',
                    'PRCP': 'float32'
                },
                engine='c'
            )
            
            # Convertir températures et précipitations
            for col in ['TAVG', 'TMIN', 'TMAX']:
                if col in df.columns:
                    df[col] = df[col] / 10.0
            
            if 'PRCP' in df.columns:
                df['PRCP'] = df['PRCP'] / 10.0
            
            return df
        except Exception as e:
            st.warning(f"⚠️ Erreur {year}: {str(e)[:50]}")
            return None
    return None

def get_data_by_years(years_requested):
    """Charge seulement les années demandées"""
    current_dir = Path(os.getcwd())
    processed_dir = current_dir / 'data_noaa' / 'processed'
    
    if not processed_dir.exists():
        st.error(f"❌ Dossier introuvable: {processed_dir}")
        return None
    
    loaded_data = []
    
    for year in years_requested:
        df_year = load_single_year(processed_dir, year)
        if df_year is not None:
            loaded_data.append(df_year)
    
    if loaded_data:
        final_df = pd.concat(loaded_data, ignore_index=True)
        st.success(f"✅ {len(loaded_data)}/{len(years_requested)} année(s) chargée(s) - {len(final_df):,} lignes")
        return final_df
    else:
        st.error("❌ Aucune donnée chargée")
        return None

def load_data_optimized():
    """Chargement OPTIMISÉ avec sélection d'années"""
    try:
        current_dir = Path(os.getcwd())
        processed_dir = current_dir / 'data_noaa' / 'processed'
        
        if not processed_dir.exists():
            return None, f"Dossier introuvable: {processed_dir}"
        
        result = {}
        files_loaded = ['annual_trends.csv']
        
        # 1. Toujours charger annual_trends (léger)
        df_trends = load_annual_trends(processed_dir)
        if df_trends is not None:
            result['annual_trends'] = df_trends

        # 2. Chargement OPTIONNEL et LIMITÉ des données détaillées
        if st.session_state.get('load_pivoted', False):
            
            # Container pour la sélection des années
            with st.sidebar:
                st.markdown("---")
                st.markdown("### 📅 Sélection des Années")
                
                # Sélection multiple avec limite
                selected_years = st.multiselect(
                    "🔍 Choisir les années à charger",
                    options=list(range(2000, 2026)),
                    default=[2023, 2024, 2025],  # 3 années par défaut
                    max_selections=8,  # Limite de sécurité
                    help="Sélectionnez maximum 8 années pour des performances optimales"
                )
                
                if not selected_years:
                    st.warning("⚠️ Veuillez sélectionner au moins une année")
                    return result, files_loaded
                
                st.info(f"📊 {len(selected_years)} année(s) sélectionnée(s)")
            
            # Chargement avec progression
            st.info(f"🔄 Chargement de {len(selected_years)} année(s)...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            loaded_dfs = []
            for i, year in enumerate(sorted(selected_years)):
                status_text.text(f"📁 Chargement {year}... ({i+1}/{len(selected_years)})")
                df_year = load_single_year(processed_dir, year)
                if df_year is not None:
                    loaded_dfs.append(df_year)
                progress_bar.progress((i + 1) / len(selected_years))
            
            if loaded_dfs:
                result['pivoted_optimized'] = pd.concat(loaded_dfs, ignore_index=True)
                files_loaded.extend([f"climate_data_pivoted_{year}.csv" for year in selected_years])
                
                # Calcul mémoire
                mem_usage = result['pivoted_optimized'].memory_usage(deep=True).sum() / 1024**2
                st.success(f"🎉 {len(loaded_dfs)} fichier(s) chargé(s) - {len(result['pivoted_optimized']):,} lignes ({mem_usage:.1f} MB)")
            
            status_text.text("✅ Chargement terminé")
        
        return result, files_loaded
    
    except Exception as e:
        st.error(f"Erreur lors du chargement: {e}")
        return None, str(e)

# ==========================================
# VISUALISATIONS
# ==========================================

def show_dashboard(data):
    """Tableau de bord climatique optimisé"""
    st.header("🎯 Tableau de Bord Climatique")
    
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
            st.dataframe(df.head(50), use_container_width=True)
    
    # KPIs
    st.subheader("📊 Indicateurs Clés de Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if cols['temp_avg']:
            try:
                avg_temp = df[cols['temp_avg']].mean()
                st.metric("🌡️ Température Moyenne", f"{avg_temp:.1f}°C")
            except:
                st.metric("🌡️ Température Moyenne", "N/A")
        else:
            st.metric("🌡️ Température Moyenne", "N/A")
    
    with col2:
        if cols['temp_max']:
            try:
                max_temp = df[cols['temp_max']].max()
                st.metric("🔥 Température Max", f"{max_temp:.1f}°C")
            except:
                st.metric("🔥 Température Max", "N/A")
        else:
            st.metric("🔥 Température Max", "N/A")
    
    with col3:
        if cols['temp_min']:
            try:
                min_temp = df[cols['temp_min']].min()
                st.metric("❄️ Température Min", f"{min_temp:.1f}°C")
            except:
                st.metric("❄️ Température Min", "N/A")
        else:
            st.metric("❄️ Température Min", "N/A")
    
    with col4:
        if cols['precip']:
            try:
                total_precip = df[cols['precip']].sum()
                st.metric("💧 Précipitations Totales", f"{total_precip:.0f} mm")
            except:
                st.metric("💧 Précipitations", "N/A")
        else:
            st.metric("💧 Précipitations", "N/A")
    
    st.markdown("---")
    
    # Graphiques
    if cols['year'] and cols['temp_avg']:
        try:
            st.subheader("📈 Évolution Temporelle des Températures")
            
            # Agréger par année
            yearly = df.groupby(cols['year']).agg({
                cols['temp_avg']: 'mean',
                cols['temp_min']: 'mean' if cols['temp_min'] else lambda x: np.nan,
                cols['temp_max']: 'mean' if cols['temp_max'] else lambda x: np.nan
            }).reset_index()
            
            fig = go.Figure()
            
            # Température moyenne
            fig.add_trace(go.Scatter(
                x=yearly[cols['year']],
                y=yearly[cols['temp_avg']],
                name='Température Moyenne',
                line=dict(color='#667eea', width=3),
                mode='lines+markers',
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)'
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
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur graphique températures: {e}")
    
    # Graphique précipitations
    if cols['year'] and cols['precip']:
        try:
            st.subheader("💧 Évolution des Précipitations")
            
            yearly_precip = df.groupby(cols['year'])[cols['precip']].sum().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=yearly_precip[cols['year']],
                y=yearly_precip[cols['precip']],
                marker_color='#3b82f6',
                name='Précipitations'
            ))
            
            fig.update_layout(
                template='plotly_white',
                height=400,
                xaxis_title="Année",
                yaxis_title="Précipitations (mm)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur graphique précipitations: {e}")

def show_comparative_analysis(data):
    """Analyse comparative optimisée"""
    st.header("🔬 Analyse Comparative Avancée")
    
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
        
        available_metrics = [c for c in [cols['temp_avg'], cols['temp_min'], cols['temp_max'], cols['precip']] if c]
        
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
            z = np.polyfit(yearly[cols['year']], yearly[metric_choice], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=yearly[cols['year']],
                y=p(yearly[cols['year']]),
                mode='lines',
                name='Tendance linéaire',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            fig.update_layout(
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            trend_value = z[0]
            st.info(f"📈 **Tendance:** {trend_value:+.4f} unités/an")
            
            if trend_value > 0:
                st.warning(f"⚠️ Augmentation de {abs(trend_value):.4f} par an sur la période")
            else:
                st.success(f"✅ Diminution de {abs(trend_value):.4f} par an sur la période")
        
        except Exception as e:
            st.error(f"Erreur: {e}")
    
    with tab2:
        st.subheader("Matrice de corrélation")
        
        numeric_cols = [c for c in [cols['temp_avg'], cols['temp_min'], cols['temp_max'], cols['precip']] if c]
        
        if len(numeric_cols) >= 2:
            try:
                corr_data = df[numeric_cols].dropna()
                corr_matrix = corr_data.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    labels=dict(color="Corrélation"),
                    zmin=-1, zmax=1
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur: {e}")
        else:
            st.warning("Pas assez de colonnes pour une matrice de corrélation")
    
    with tab3:
        st.subheader("Analyse saisonnière")
        
        if cols['month'] and cols['temp_avg']:
            try:
                monthly = df.groupby(cols['month'])[cols['temp_avg']].mean().reset_index()
                
                month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                               'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[month_names[int(m)-1] for m in monthly[cols['month']]],
                    y=monthly[cols['temp_avg']],
                    marker_color='#667eea',
                    text=monthly[cols['temp_avg']].round(1),
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Température moyenne par mois",
                    template='plotly_white',
                    height=400,
                    xaxis_title="Mois",
                    yaxis_title="Température (°C)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur: {e}")
        else:
            st.info("💡 Activez le chargement des données détaillées pour voir l'analyse saisonnière")

def show_data_explorer(data):
    """Explorateur de données optimisé"""
    st.header("🔍 Explorateur de Données Interactif")
    
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
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        n_rows = st.slider("Nombre de lignes à afficher", 10, 1000, 100, 10)
    with col_opt2:
        auto_refresh = st.checkbox("Actualisation automatique", value=False)
    
    st.dataframe(df.head(n_rows), use_container_width=True, height=400)
    
    # Statistiques rapides
    with st.expander("📈 Statistiques rapides"):
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
# INTERFACE PRINCIPALE OPTIMISÉE
# ==========================================

def main():
    # Initialisation session state
    if 'load_pivoted' not in st.session_state:
        st.session_state.load_pivoted = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    st.title("🌍 Plateforme d'Analyse Climatique Intelligente")
    st.markdown("### 🚀 Données NOAA 2000-2025 - Visualisation & Analyse Avancée")
    
    with st.sidebar:
        st.title("🎛️ Configuration")
        
        # SOURCE DE DONNÉES OPTIMISÉE
        st.markdown("### 📂 Source de données")
        
        if not st.session_state.load_pivoted:
            st.success("✅ Mode léger activé")
            st.info("""
            **Données chargées:**
            - 📊 `annual_trends.csv` seulement
            - ⚡ Performances optimales
            - 📈 Analyses agrégées
            """)
            
            if st.button("🔄 ACTIVER DONNÉES DÉTAILLÉES", type="primary", use_container_width=True):
                st.session_state.load_pivoted = True
                st.session_state.data_loaded = False
                st.experimental_rerun()
        else:
            st.warning("🚨 Mode données détaillées")
            st.info("""
            **Fonctionnalités activées:**
            - 📅 Sélection d'années spécifiques
            - 🔍 Analyses journalières
            - 📊 Visualisations avancées
            """)
            
            if st.button("⬅️ MODE LÉGER", type="secondary", use_container_width=True):
                st.session_state.load_pivoted = False
                st.session_state.data_loaded = False
                st.experimental_rerun()
        
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
        
        # INFORMATION PERFORMANCE
        st.markdown("---")
        st.markdown("### 💡 Conseils Performance")
        st.info("""
        - 🎯 **Mode léger**: Idéal pour démarrage
        - 📅 **Max 8 années**: Pour stabilité
        - 💾 **Mémoire**: Surveillez l'usage
        - 🔄 **Rechargez** si nécessaire
        """)
    
    # Chargement OPTIMISÉ
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Chargement optimisé en cours..."):
            data, info = load_data_optimized()
        
        if data is not None:
            st.session_state.data = data
            st.session_state.data_loaded = True
            st.session_state.load_info = info
        else:
            st.error(f"❌ Erreur de chargement: {info}")
            return
    else:
        data = st.session_state.data
        info = st.session_state.load_info
    
    # Info données chargées
    with st.expander("ℹ️ Informations de chargement", expanded=False):
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
        
        st.metric("💾 **Mémoire totale utilisée**", f"{total_memory:.1f} MB")
        
        if st.button("🔄 Recharger les données"):
            st.session_state.data_loaded = False
            st.experimental_rerun()
    
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