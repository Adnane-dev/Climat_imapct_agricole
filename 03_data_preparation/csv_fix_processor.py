import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class ClimateRegressionModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        self.features = []
        self.target = 'TMAX'
    
    def prepare_data(self, data_path):
        """Prépare les données pour la régression"""
        try:
            df = pd.read_parquet(data_path)
            
            # Créer des features de base si elles n'existent pas
            if 'DATE' in df.columns:
                df['DATE'] = pd.to_datetime(df['DATE'])
                df['month'] = df['DATE'].dt.month
                df['day_of_year'] = df['DATE'].dt.dayofyear
                df['season'] = (df['month'] % 12 // 3) + 1
            
            # Features pour la prédiction de température
            feature_columns = [
                'TMIN', 'PRCP', 'TAVG', 'month', 'season', 'day_of_year'
            ]
            
            # Ajouter des features de rolling si possible
            if len(df) > 1000:
                try:
                    df['TMAX_rolling_mean_7'] = df.groupby('ID')['TMAX'].transform(
                        lambda x: x.rolling(7, min_periods=1).mean()
                    )
                    df['TMAX_rolling_std_7'] = df.groupby('ID')['TMAX'].transform(
                        lambda x: x.rolling(7, min_periods=1).std()
                    )
                    feature_columns.extend(['TMAX_rolling_mean_7', 'TMAX_rolling_std_7'])
                except:
                    pass
            
            # Nettoyage
            available_features = [f for f in feature_columns if f in df.columns]
            df_clean = df.dropna(subset=available_features + [self.target])
            
            if len(df_clean) < 100:
                raise ValueError("Pas assez de données après nettoyage")
            
            self.features = available_features
            X = df_clean[available_features]
            y = df_clean[self.target]
            
            return X, y, df_clean
            
        except Exception as e:
            print(f"❌ Erreur préparation données: {e}")
            # Retourner des données simulées pour la démo
            return self.create_demo_data()
    
    def create_demo_data(self):
        """Crée des données de démonstration"""
        print("🔄 Création de données de démonstration...")
        np.random.seed(42)
        
        n_samples = 1000
        X_demo = pd.DataFrame({
            'TMIN': np.random.normal(10, 5, n_samples),
            'PRCP': np.random.exponential(5, n_samples),
            'TAVG': np.random.normal(15, 4, n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'season': np.random.randint(1, 5, n_samples),
            'day_of_year': np.random.randint(1, 366, n_samples),
            'TMAX_rolling_mean_7': np.random.normal(20, 3, n_samples),
            'TMAX_rolling_std_7': np.random.exponential(2, n_samples)
        })
        
        # Relation simulée: TMAX ≈ 1.2*TAVG + 0.5*TMIN + bruit
        y_demo = 1.2 * X_demo['TAVG'] + 0.5 * X_demo['TMIN'] + np.random.normal(0, 2, n_samples)
        
        self.features = X_demo.columns.tolist()
        return X_demo, y_demo, X_demo
    
    def train_model(self, X, y):
        """Entraîne le modèle de régression"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("🧠 Entraînement du modèle de régression...")
        self.model.fit(X_train, y_train)
        
        # Évaluation
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Performance du modèle - RMSE: {rmse:.2f}°C, R²: {r2:.2f}")
        
        return X_test, y_test, y_pred
    
    def save_model(self, path="../04_modeling/models_saved/regression_model.pkl"):
        """Sauvegarde le modèle entraîné"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'features': self.features,
            'target': self.target,
            'metadata': {
                'model_type': 'RandomForestRegressor',
                'version': '1.0',
                'training_date': pd.Timestamp.now().strftime('%Y-%m-%d')
            }
        }
        
        joblib.dump(model_data, path)
        print(f"💾 Modèle sauvegardé: {path}")

def train_regression_model():
    """Fonction pour entraîner le modèle de régression"""
    print("🚀 Lancement de l'entraînement du modèle de régression...")
    
    regressor = ClimateRegressionModel()
    
    # Essayer de charger les données réelles
    data_path = "../03_data_preparation/data_noaa/processed/massive_dataset.parquet"
    if Path(data_path).exists():
        X, y, df = regressor.prepare_data(data_path)
    else:
        print("📁 Dataset massif non trouvé, utilisation données de démonstration")
        X, y, df = regressor.prepare_data(None)
    
    # Entraînement
    X_test, y_test, y_pred = regressor.train_model(X, y)
    
    # Sauvegarde
    regressor.save_model()
    
    return regressor

if __name__ == "__main__":
    train_regression_model()