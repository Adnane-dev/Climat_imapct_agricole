import pandas as pd
import glob
import os
from pathlib import Path

def merge_pivoted_files(input_dir='../data_noaa/processed', output_file='climate_data_pivoted.csv'):
    """
    Fusionne tous les fichiers climatiques pivotés en un seul fichier
    """
    print("="*60)
    print("🌍 FUSION DES DONNÉES CLIMATIQUES PIVOTÉES")
    print("="*60)
    
    # Créer le répertoire de sortie si nécessaire
    output_dir = os.path.dirname(os.path.join(input_dir, output_file))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Trouver tous les fichiers pivotés
    pattern = os.path.join(input_dir, "climate_data_pivoted_*.csv")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"❌ Aucun fichier pivoté trouvé avec le motif: {pattern}")
        return None
        
    print(f"📁 {len(files)} fichiers pivotés trouvés")
    
    # Variables pour suivre la progression
    total_rows = 0
    skipped_rows = 0
    dfs = []
    
    # Traiter chaque fichier
    for file in files:
        filename = os.path.basename(file)
        year = filename.split('_')[-1].split('.')[0]
        print(f"\nTraitement de l'année {year}...")
        
        try:
            # Lire le fichier avec des types optimisés
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
                on_bad_lines='skip',  # Ignorer les lignes problématiques
                verbose=True         # Afficher les détails des lignes ignorées
            )
            
            # Vérifier la cohérence des données
            if 'YEAR' in df.columns and df['YEAR'].nunique() == 1:
                if int(year) != df['YEAR'].iloc[0]:
                    print(f"⚠️ Attention: L'année du fichier ({year}) ne correspond pas aux données ({df['YEAR'].iloc[0]})")
            
            total_rows += len(df)
            dfs.append(df)
            print(f"✓ {filename}: {len(df):,} lignes chargées")
            
        except Exception as e:
            print(f"⚠️ Erreur avec {filename}: {str(e)}")
            print(f"  Type d'erreur: {type(e).__name__}")
            continue
    
    if not dfs:
        print("\n❌ Aucune donnée n'a pu être chargée!")
        return None
    
    # Fusionner tous les DataFrames
    print("\nFusion des données...")
    final_df = pd.concat(dfs, ignore_index=True)
    print(f"✓ {total_rows:,} lignes au total")
    
    # Sauvegarder le résultat
    output_path = os.path.join(input_dir, output_file)
    print(f"\nSauvegarde dans {output_file}...")
    final_df.to_csv(output_path, index=False)
    print("✓ Sauvegarde terminée")
    
    # Afficher un résumé
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES DONNÉES")
    print("="*60)
    print(f"📆 Période : {final_df['YEAR'].min()}-{final_df['YEAR'].max()}")
    print(f"📍 Stations : {final_df['ID'].nunique():,}")
    print(f"📝 Observations : {len(final_df):,}")
    
    return final_df

if __name__ == "__main__":
    merge_pivoted_files()