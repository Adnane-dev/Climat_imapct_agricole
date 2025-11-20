import pandas as pd
import numpy as np
from datetime import datetime

class ClimateFeatureEngineer:
    def __init__(self):
        self.features = []
    
    def create_temporal_features(self, df, date_column='DATE'):
        """Crée les features temporelles"""
        df[date_column] = pd.to_datetime(df[date_column])
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['season'] = df['month'] % 12 // 3 + 1
        df['day_of_year'] = df[date_column].dt.dayofyear
        
        return df
    
    def create_rolling_features(self, df, column='TMAX', window=7):
        """Features de moyennes glissantes"""
        df[f'{column}_rolling_mean_7'] = df.groupby('STATION')[column].rolling(window=window).mean().reset_index(0, drop=True)
        df[f'{column}_rolling_std_7'] = df.groupby('STATION')[column].rolling(window=window).std().reset_index(0, drop=True)
        
        return df
    
    def create_climate_indices(self, df):
        """Indices climatiques pour l'agriculture"""
        # Indice de stress thermique
        df['heat_stress'] = np.where(df['TMAX'] > 30, df['TMAX'] - 30, 0)
        
        # Indice de sécheresse (simplifié)
        if 'PRCP' in df.columns:
            df['drought_index'] = df.groupby(['STATION', 'year'])['PRCP'].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
        
        return df
    
    def engineer_all_features(self, df):
        """Pipeline complet de feature engineering"""
        print("Début du feature engineering...")
        
        # Features temporelles
        df = self.create_temporal_features(df)
        
        # Features de rolling
        numeric_cols = ['TMAX', 'TMIN', 'PRCP', 'TAVG']
        for col in numeric_cols:
            if col in df.columns:
                df = self.create_rolling_features(df, col)
        
        # Indices climatiques
        df = self.create_climate_indices(df)
        
        print(f"Feature engineering terminé. {len(df.columns)} colonnes créées.")
        return df