import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

# ==========================================
# 1. FUSION DES 25 DATASETS CSV (2000-2025)
# ==========================================

def merge_climate_datasets(input_folder='../data_noaa/processed', output_file='climate_data_pivoted.csv'):
    """
    Fusionne tous les fichiers CSV climatiques pivotés en un seul dataset
    
    Args:
        input_folder: Dossier contenant les CSV (ex: "data_noaa/processed/")
        output_file: Nom du fichier de sortie
    """
    print("🔄 Début de la fusion des datasets...")
    print(f"📂 Dossier d'entrée: {input_folder}")
    
    # Récupérer tous les fichiers CSV pivotés
    csv_files = glob.glob(os.path.join(input_folder, "climate_data_pivoted_*.csv"))
    
    if not csv_files:
        print("❌ Aucun fichier CSV trouvé dans le dossier")
        print(f"🔍 Motif recherché: {os.path.join(input_folder, 'climate_data_pivoted_*.csv')}")
        return None
    
    print(f"📁 {len(csv_files)} fichiers CSV trouvés")
    
    # Liste pour stocker les dataframes
    dfs = []
    
    for file in csv_files:
        try:
            year = os.path.basename(file).split('_')[-1].split('.')[0]
            print(f"\nTraitement de {os.path.basename(file)}...")
            
            # Lire chaque CSV avec des types de données optimisés
            df = pd.read_csv(
                file,
                dtype={
                    'ID': 'category',
                    'YEAR': 'int16',
                    'MONTH': 'int8',
                    'DAY': 'int8',
                    'TAVG': 'float32',
                    'TMIN': 'float32',
                    'TMAX': 'float32',
                    'PRCP': 'float32'
                },
                skipinitialspace=True,
                on_bad_lines='skip',   # Ignorer les lignes problématiques
                verbose=True          # Afficher les détails des lignes ignorées
            )
            
            # Renommer les colonnes pour standardiser
            df.columns = df.columns.str.strip().str.upper()
            
            # Ajouter l'année du fichier si nécessaire
            filename = os.path.basename(file)
            year = extract_year_from_filename(filename)
            if year and 'YEAR' not in df.columns:
                df['YEAR'] = year
            
            dfs.append(df)
            print(f"✅ {filename}: {len(df)} lignes chargées")
            
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de {file}: {e}")
    
    # Fusionner tous les dataframes
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Sauvegarder
        merged_df.to_csv(output_file, index=False)
        print(f"\n✅ Fusion terminée: {len(merged_df)} lignes totales")
        print(f"📊 Fichier sauvegardé: {output_file}")
        
        return merged_df
    
    return None


def extract_year_from_filename(filename):
    """Extrait l'année du nom de fichier"""
    import re
    match = re.search(r'(20\d{2})', filename)
    return int(match.group(1)) if match else None


# ==========================================
# 2. NETTOYAGE ET PRÉPARATION DES DONNÉES
# ==========================================

def clean_climate_data(df):
    """
    Nettoie et prépare les données climatiques
    
    Structure attendue:
    - ID: Identifiant de la station
    - DATE: Date au format YYYYMMDD
    - ELEMENT: Type de mesure (TAVG, PRCP, TMAX, etc.)
    - DATA_VALUE: Valeur de la mesure
    - M_FLAG, Q_FLAG, S_FLAG: Flags de qualité
    - OBS_TIME: Heure d'observation
    """
    print("\n🧹 Nettoyage des données...")
    
    # Copie pour éviter les modifications en place
    df = df.copy()
    
    # 1. Supprimer les espaces dans les colonnes
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    
    # 2. Convertir DATE en datetime
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d', errors='coerce')
        df['YEAR'] = df['DATE'].dt.year
        df['MONTH'] = df['DATE'].dt.month
        df['DAY'] = df['DATE'].dt.day
    
    # 3. Filtrer les données de qualité (Q_FLAG vide ou acceptables)
    if 'Q_FLAG' in df.columns:
        df = df[df['Q_FLAG'].isna() | (df['Q_FLAG'] == '')]
    
    # 4. Gérer les valeurs manquantes
    df = df.dropna(subset=['ID', 'DATE', 'ELEMENT', 'DATA_VALUE'])
    
    # 5. Convertir DATA_VALUE en numérique
    df['DATA_VALUE'] = pd.to_numeric(df['DATA_VALUE'], errors='coerce')
    
    print(f"✅ Données nettoyées: {len(df)} lignes")
    
    return df


# ==========================================
# 3. ENRICHISSEMENT AVEC DONNÉES STATIONS
# ==========================================

def load_station_metadata(station_file='ghcnd-stations.txt'):
    """
    Charge les métadonnées des stations depuis ghcnd-stations.txt
    
    Format: ID LATITUDE LONGITUDE ELEVATION STATE NAME
    """
    print("\n📍 Chargement des métadonnées des stations...")
    
    try:
        # Colonnes fixes du fichier stations
        colspecs = [
            (0, 11),    # ID
            (12, 20),   # LATITUDE
            (21, 30),   # LONGITUDE
            (31, 37),   # ELEVATION
            (38, 40),   # STATE
            (41, 71)    # NAME
        ]
        
        stations = pd.read_fwf(
            station_file,
            colspecs=colspecs,
            names=['ID', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'STATE', 'NAME']
        )
        
        # Nettoyer les espaces
        stations['ID'] = stations['ID'].str.strip()
        stations['NAME'] = stations['NAME'].str.strip()
        
        print(f"✅ {len(stations)} stations chargées")
        
        return stations
        
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement des stations: {e}")
        return None


def enrich_with_stations(climate_df, stations_df):
    """Enrichit les données climatiques avec les informations géographiques"""
    print("\n🔗 Enrichissement avec données géographiques...")
    
    enriched = climate_df.merge(
        stations_df[['ID', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME']],
        on='ID',
        how='left'
    )
    
    print(f"✅ {len(enriched)} lignes enrichies")
    
    return enriched


# ==========================================
# 4. PIVOT DES DONNÉES POUR ANALYSE
# ==========================================

def pivot_climate_data(df):
    """
    Pivote les données pour avoir une colonne par élément climatique
    Transforme de format long à format large
    """
    print("\n🔄 Pivot des données...")
    
    # Créer un pivot avec DATE et ID comme index
    pivoted = df.pivot_table(
        index=['DATE', 'ID', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME', 'YEAR', 'MONTH'],
        columns='ELEMENT',
        values='DATA_VALUE',
        aggfunc='mean'  # Moyenne si plusieurs valeurs
    ).reset_index()
    
    print(f"✅ Données pivotées: {len(pivoted)} lignes, {len(pivoted.columns)} colonnes")
    
    return pivoted


# ==========================================
# 5. CRÉATION DE RÉGIONS GÉOGRAPHIQUES
# ==========================================

def assign_regions(df):
    """
    Assigne des régions basées sur la latitude/longitude
    Adapté pour la Tunisie ou autres pays
    """
    print("\n🗺️ Attribution des régions...")
    
    df = df.copy()
    
    # Définir les régions (à adapter selon votre zone géographique)
    def get_region(lat, lon):
        if pd.isna(lat) or pd.isna(lon):
            return 'Inconnu'
        
        # Exemple pour la Tunisie
        if lat > 36:
            return 'Nord'
        elif lat > 34:
            return 'Centre'
        else:
            return 'Sud'
    
    df['REGION'] = df.apply(lambda row: get_region(row['LATITUDE'], row['LONGITUDE']), axis=1)
    
    print(f"✅ Régions attribuées: {df['REGION'].value_counts().to_dict()}")
    
    return df


# ==========================================
# 6. CALCUL D'INDICATEURS AGRICOLES
# ==========================================

def calculate_agricultural_indicators(df):
    """
    Calcule des indicateurs pertinents pour l'agriculture
    """
    print("\n🌾 Calcul des indicateurs agricoles...")
    
    df = df.copy()
    
    # Growing Degree Days (GDD) - Base 10°C
    if 'TAVG' in df.columns:
        df['GDD'] = df['TAVG'].apply(lambda x: max(0, (x/10) - 10) if pd.notna(x) else 0)
    
    # Stress hydrique (jours sans pluie)
    if 'PRCP' in df.columns:
        df['DRY_DAY'] = (df['PRCP'] == 0).astype(int)
    
    # Risque de gel
    if 'TMIN' in df.columns:
        df['FROST_RISK'] = (df['TMIN'] < 0).astype(int)
    
    # Vague de chaleur
    if 'TMAX' in df.columns:
        df['HEAT_WAVE'] = (df['TMAX'] > 350).astype(int)  # >35°C
    
    # Pluie excessive
    if 'PRCP' in df.columns:
        df['HEAVY_RAIN'] = (df['PRCP'] > 500).astype(int)  # >50mm
    
    print("✅ Indicateurs calculés: GDD, DRY_DAY, FROST_RISK, HEAT_WAVE, HEAVY_RAIN")
    
    return df


# ==========================================
# 7. AGRÉGATIONS POUR VISUALISATION
# ==========================================

def create_aggregations(df):
    """
    Crée différentes agrégations pour faciliter la visualisation
    """
    print("\n📊 Création des agrégations...")
    
    aggregations = {}
    
    # 1. Agrégation annuelle par région
    annual_regional = df.groupby(['YEAR', 'REGION']).agg({
        'TAVG': 'mean',
        'TMAX': 'mean',
        'TMIN': 'mean',
        'PRCP': 'sum',
        'GDD': 'sum',
        'DRY_DAY': 'sum',
        'FROST_RISK': 'sum',
        'HEAT_WAVE': 'sum'
    }).reset_index()
    
    aggregations['annual_regional'] = annual_regional
    print(f"  ✅ Agrégation annuelle/régionale: {len(annual_regional)} lignes")
    
    # 2. Agrégation mensuelle
    monthly = df.groupby(['YEAR', 'MONTH']).agg({
        'TAVG': 'mean',
        'PRCP': 'sum',
        'TMAX': 'max',
        'TMIN': 'min'
    }).reset_index()
    
    aggregations['monthly'] = monthly
    print(f"  ✅ Agrégation mensuelle: {len(monthly)} lignes")
    
    # 3. Agrégation par station (moyenne sur toute la période)
    station_avg = df.groupby(['ID', 'NAME', 'LATITUDE', 'LONGITUDE', 'REGION']).agg({
        'TAVG': 'mean',
        'PRCP': 'mean',
        'TMAX': 'mean',
        'TMIN': 'mean'
    }).reset_index()
    
    aggregations['station_avg'] = station_avg
    print(f"  ✅ Agrégation par station: {len(station_avg)} lignes")
    
    # 4. Tendances annuelles globales
    annual_trends = df.groupby('YEAR').agg({
        'TAVG': 'mean',
        'TMAX': 'mean',
        'TMIN': 'mean',
        'PRCP': 'sum',
        'GDD': 'sum',
        'HEAT_WAVE': 'sum',
        'FROST_RISK': 'sum',
        'DRY_DAY': 'sum'
    }).reset_index()
    
    aggregations['annual_trends'] = annual_trends
    print(f"  ✅ Tendances annuelles: {len(annual_trends)} lignes")
    
    return aggregations


# ==========================================
# 8. FONCTION PRINCIPALE
# ==========================================

def process_climate_data(
    input_folder='../data_noaa',
    station_file='../data_noaa/ghcnd-stations.txt',
    output_folder='../data_noaa/processed'
):
    """
    Pipeline complet de traitement des données climatiques
    """
    print("="*60)
    print("🌍 TRAITEMENT DES DONNÉES CLIMATIQUES 2000-2025")
    print("="*60)
    
    # Créer le dossier de sortie
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 1. Fusion des datasets
    merged_file = os.path.join(output_folder, 'merged_climate_data.csv')
    climate_df = merge_climate_datasets(input_folder, merged_file)
    
    if climate_df is None:
        print("❌ Échec de la fusion")
        return None
    
    # 2. Nettoyage
    climate_df = clean_climate_data(climate_df)
    
    # 3. Chargement des stations
    stations_df = load_station_metadata(station_file)
    
    # 4. Enrichissement
    if stations_df is not None:
        climate_df = enrich_with_stations(climate_df, stations_df)
    
    # 5. Pivot
    pivoted_df = pivot_climate_data(climate_df)
    
    # 6. Attribution des régions
    pivoted_df = assign_regions(pivoted_df)
    
    # 7. Calcul des indicateurs
    pivoted_df = calculate_agricultural_indicators(pivoted_df)
    
    # 8. Sauvegarde des données pivotées
    pivoted_file = os.path.join(output_folder, 'climate_data_pivoted.csv')
    pivoted_df.to_csv(pivoted_file, index=False)
    print(f"\n💾 Données pivotées sauvegardées: {pivoted_file}")
    
    # 9. Création des agrégations
    aggregations = create_aggregations(pivoted_df)
    
    # 10. Sauvegarde des agrégations
    for name, agg_df in aggregations.items():
        agg_file = os.path.join(output_folder, f'{name}.csv')
        agg_df.to_csv(agg_file, index=False)
        print(f"💾 {name} sauvegardé: {agg_file}")
    
    print("\n" + "="*60)
    print("✅ TRAITEMENT TERMINÉ AVEC SUCCÈS")
    print("="*60)
    
    return {
        'full_data': pivoted_df,
        'aggregations': aggregations
    }


# ==========================================
# 9. ANALYSE DES DONNÉES DE COLONNES
# ==========================================

def analyze_dataset(csv_file):
    """
    Analyse exploratoire d'un dataset
    """
    print(f"\n📊 Analyse de {csv_file}")
    print("="*60)
    
    df = pd.read_csv(csv_file)
    
    print(f"\n📏 Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"\n📋 Colonnes disponibles:\n{df.columns.tolist()}")
    print(f"\n🔢 Types de données:\n{df.dtypes}")
    print(f"\n📊 Statistiques descriptives:\n{df.describe()}")
    print(f"\n❓ Valeurs manquantes:\n{df.isnull().sum()}")
    
    if 'ELEMENT' in df.columns:
        print(f"\n🌡️ Éléments climatiques disponibles:\n{df['ELEMENT'].value_counts()}")
    
    if 'YEAR' in df.columns or 'DATE' in df.columns:
        date_col = 'YEAR' if 'YEAR' in df.columns else 'DATE'
        print(f"\n📅 Période couverte: {df[date_col].min()} - {df[date_col].max()}")
    
    return df


# ==========================================
# EXEMPLE D'UTILISATION
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("🌍 TRAITEMENT DES DONNÉES CLIMATIQUES 2000-2025")
    print("=" * 60)
    
    # Fusion des données pivotées
    merged_data = merge_climate_datasets()
    
    if merged_data is not None:
        # Analyse rapide des données fusionnées
        print("\n" + "="*60)
        print("📈 RÉSUMÉ DES DONNÉES FUSIONNÉES")
        print("="*60)
        
        print(f"\n✅ Observations totales : {len(merged_data):,}")
        print(f"✅ Période : {merged_data['YEAR'].min()}-{merged_data['YEAR'].max()}")
        print(f"✅ Stations : {merged_data['ID'].nunique():,}")
        
        # Variables climatiques disponibles
        climate_vars = [col for col in merged_data.columns if col not in 
                       ['DATE', 'ID', 'YEAR', 'MONTH', 'DAY']]
        print(f"\n🌡️ Variables climatiques : {', '.join(climate_vars)}")
        print("\n✅ Fusion terminée avec succès!")
