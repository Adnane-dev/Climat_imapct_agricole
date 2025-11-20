import pandas as pd
import pyarrow.parquet as pq
import os
from climate_data_processor import ClimateDataProcessor  # Votre code existant
from feature_engineering import ClimateFeatureEngineer

class MassiveDatasetBuilder:
    def __init__(self, data_path="../02_data_understanding/data_noaa/raw/"):
        self.data_path = data_path
        self.processor = ClimateDataProcessor()
        self.feature_engineer = ClimateFeatureEngineer()
    
    def build_massive_dataset(self, target_size=10000000):
        """Construit le dataset massif de 10M+ points"""
        print("Construction du dataset massif...")
        
        all_data = []
        years = range(2000, 2026)
        
        for year in years:
            file_path = f"{self.data_path}/{year}.csv"
            if os.path.exists(file_path):
                print(f"Traitement de {file_path}...")
                
                # Utilisez votre processeur existant
                df_year = self.processor.load_and_process_year(file_path)
                all_data.append(df_year)
                
                # Vérification taille cumulative
                total_size = sum(len(df) for df in all_data)
                print(f"Taille cumulative: {total_size:,} points")
                
                if total_size >= target_size:
                    break
        
        # Concaténation
        massive_df = pd.concat(all_data, ignore_index=True)
        
        # Feature engineering
        massive_df = self.feature_engineer.engineer_all_features(massive_df)
        
        # Sauvegarde en Parquet (format optimisé Big Data)
        output_path = "../03_data_preparation/data_noaa/processed/massive_dataset.parquet"
        massive_df.to_parquet(output_path, index=False)
        print(f"Dataset massif sauvegardé: {output_path}")
        print(f"Taille finale: {len(massive_df):,} points de données")
        
        return massive_df

if __name__ == "__main__":
    builder = MassiveDatasetBuilder()
    massive_data = builder.build_massive_dataset()