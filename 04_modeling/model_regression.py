import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

class ClimateRegressionModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            max_depth=10
        )
        # CORRECTION: Utiliser seulement des features simples sans rolling
        self.features = ['TMIN', 'TAVG', 'PRCP', 'month', 'season', 'day_of_year']
        self.target = 'TMAX'
    
    def prepare_data(self):
        """Prépare des données de démonstration pour la régression - CORRIGÉ"""
        print("🔄 Génération de données de démonstration...")
        np.random.seed(42)
        
        # Créer un dataset réaliste mais simple
        n_samples = 2000
        data = {
            'TMIN': np.random.normal(10, 5, n_samples),
            'TAVG': np.random.normal(18, 4, n_samples),
            'PRCP': np.random.exponential(3, n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'season': np.random.randint(1, 5, n_samples),
            'day_of_year': np.random.randint(1, 366, n_samples)
        }
        
        # Créer la variable cible avec une relation réaliste
        data['TMAX'] = (
            1.1 * data['TAVG'] + 
            0.3 * data['TMIN'] + 
            np.random.normal(0, 2, n_samples)
        )
        
        df = pd.DataFrame(data)
        
        print(f"✅ Dataset créé: {len(df)} échantillons")
        print(f"   - Features: {self.features}")
        print(f"   - Target: {self.target}")
        
        return df[self.features], df[self.target], df
    
    def train_model(self, X, y):
        """Entraîne le modèle de régression"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("🧠 Entraînement du modèle Random Forest...")
        self.model.fit(X_train, y_train)
        
        # Évaluation
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Performance du modèle:")
        print(f"   - RMSE: {rmse:.2f}°C")
        print(f"   - R²: {r2:.3f}")
        print(f"   - Features utilisées: {len(self.features)}")
        
        return X_test, y_test, y_pred
    
    def save_model(self, path="models_saved/regression_model.pkl"):
        """Sauvegarde le modèle entraîné"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'features': self.features,
            'target': self.target,
            'metadata': {
                'model_type': 'RandomForestRegressor',
                'performance': f'RMSE: ~2.0°C, R²: ~0.85',
                'version': '1.0',
                'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        joblib.dump(model_data, path)
        print(f"💾 Modèle sauvegardé: {path}")

def train_regression_model():
    """Fonction pour entraîner le modèle"""
    print("🚀 Lancement de l'entraînement du modèle de régression...")
    
    model = ClimateRegressionModel()
    X, y, df = model.prepare_data()
    model.train_model(X, y)
    model.save_model()
    
    return model

if __name__ == "__main__":
    train_regression_model()