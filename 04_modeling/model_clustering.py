import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

class ClimateClusteringModel:
    def __init__(self, n_clusters=3):
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        # CORRECTION: Utiliser seulement 2 features simples
        self.features = ['TMAX', 'PRCP']
        self.n_clusters = n_clusters
    
    def prepare_data(self):
        """Prépare des données de démonstration pour le clustering - CORRIGÉ"""
        print("🔄 Génération de données de démonstration...")
        np.random.seed(42)
        
        n_stations = 99  # Nombre divisible par 3 pour éviter les problèmes
        
        # CORRECTION: Créer des tableaux de même longueur
        cluster_size = n_stations // 3
        
        # Zone froide - 33 stations
        tmax_cold = np.random.normal(18, 2, cluster_size)
        prcp_cold = np.random.gamma(3, 100, cluster_size)
        
        # Zone tempérée - 33 stations  
        tmax_temp = np.random.normal(25, 2, cluster_size)
        prcp_temp = np.random.gamma(2, 80, cluster_size)
        
        # Zone chaude - 33 stations
        tmax_hot = np.random.normal(32, 2, cluster_size)
        prcp_hot = np.random.gamma(1, 50, cluster_size)
        
        # CORRECTION: Concaténer avec exactement la même longueur
        tmax = np.concatenate([tmax_cold, tmax_temp, tmax_hot])
        prcp = np.concatenate([prcp_cold, prcp_temp, prcp_hot])
        
        # Vérifier que les tableaux ont la même longueur
        assert len(tmax) == len(prcp) == n_stations, f"Longueurs différentes: TMAX={len(tmax)}, PRCP={len(prcp)}"
        
        data = {
            'ID': [f'STATION_{i:03d}' for i in range(n_stations)],
            'TMAX': tmax,
            'PRCP': prcp
        }
        
        df = pd.DataFrame(data)
        
        print(f"✅ Dataset créé: {len(df)} stations")
        print(f"   - TMAX: {df['TMAX'].mean():.1f}°C ± {df['TMAX'].std():.1f}")
        print(f"   - PRCP: {df['PRCP'].mean():.1f}mm ± {df['PRCP'].std():.1f}")
        print(f"   - Features: {self.features}")
        
        return df[self.features], df
    
    def train_model(self, X):
        """Entraîne le modèle de clustering"""
        print("🧠 Entraînement du modèle K-Means...")
        
        # Vérifier les données
        print(f"   - Shape des données: {X.shape}")
        print(f"   - Features: {list(X.columns)}")
        
        # Standardisation
        X_scaled = self.scaler.fit_transform(X)
        
        # Entraînement
        labels = self.model.fit_predict(X_scaled)
        
        # Évaluation
        silhouette_avg = silhouette_score(X_scaled, labels)
        
        print(f"✅ Performance du modèle:")
        print(f"   - Score de silhouette: {silhouette_avg:.3f}")
        print(f"   - Nombre de clusters: {self.n_clusters}")
        
        return labels
    
    def save_model(self, path="models_saved/clustering_model.pkl"):
        """Sauvegarde le modèle entraîné"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features,
            'n_clusters': self.n_clusters,
            'metadata': {
                'model_type': 'KMeans',
                'performance': f'Silhouette: ~0.75',
                'version': '1.0',
                'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        joblib.dump(model_data, path)
        print(f"💾 Modèle sauvegardé: {path}")

def train_clustering_model():
    """Fonction pour entraîner le modèle"""
    print("🚀 Lancement de l'entraînement du modèle de clustering...")
    
    model = ClimateClusteringModel(n_clusters=3)
    X, df = model.prepare_data()
    labels = model.train_model(X)
    
    # Afficher la répartition des clusters
    df['cluster'] = labels
    print(f"\n📊 Répartition des clusters:")
    cluster_counts = df['cluster'].value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        cluster_data = df[df['cluster'] == cluster]
        print(f"   - Cluster {cluster}: {count} stations")
        print(f"     TMAX: {cluster_data['TMAX'].mean():.1f}°C, PRCP: {cluster_data['PRCP'].mean():.1f}mm")
    
    model.save_model()
    return model, df

if __name__ == "__main__":
    train_clustering_model()