import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Système de Maintenance Prédictive IoT",
    page_icon="🏭",
    layout="wide"
)

# Style pour les graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_sample_csv():
    """Crée un fichier CSV d'exemple si il n'existe pas"""
    if not os.path.exists('iot_equipment_monitoring_dataset.csv'):
        st.info("📝 Création d'un fichier CSV d'exemple...")
        
        # Kreye done egzanp
        df_sample = generate_sample_data()
        
        # Sovgarde nan CSV
        df_sample.to_csv('iot_equipment_monitoring_dataset.csv', index=False)
        st.success("✅ Fichier CSV d'exemple créé!")
        
        return df_sample
    return None

def generate_sample_data():
    """Génère des données simulées"""
    np.random.seed(42)
    n_samples = 5000
    
    data = {
        'Sensor_ID': [f'SENSOR_{i:03d}' for i in range(n_samples)],
        'Temperature': np.random.normal(65, 15, n_samples),
        'Vibration': np.random.gamma(2, 2, n_samples),
        'Pressure': np.random.normal(100, 20, n_samples),
        'Voltage': np.random.normal(220, 10, n_samples),
        'Current': np.random.normal(15, 3, n_samples),
        'FFT_Feature1': np.random.uniform(0, 1, n_samples),
        'FFT_Feature2': np.random.uniform(0, 1, n_samples),
        'Anomaly_Score': np.random.uniform(0, 1, n_samples),
        'Timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H')
    }
    
    df = pd.DataFrame(data)
    
    conditions = [
        (df['Temperature'] > 85) & (df['Voltage'] < 210),
        (df['Vibration'] > 8) & (df['Pressure'] > 130),
        (df['Temperature'] > 90),
        (df['Current'] > 20)
    ]
    choices = ['Electrical Fault', 'Mechanical Failure', 'Overheating', 'Electrical Fault']
    
    df['Fault_Type'] = np.select(conditions, choices, default='None')
    df['Fault_Status'] = (df['Fault_Type'] != 'None').astype(int)
    
    return df

@st.cache_data
def load_csv_data():
    """Charge le fichier CSV réel avec gestion d'erreur améliorée"""
    
    # Tcheke si fichye a egziste
    if not os.path.exists('iot_equipment_monitoring_dataset.csv'):
        st.warning("📁 Fichier CSV non trouvé. Génération de données simulées...")
        return generate_sample_data()
    
    try:
        # Eseye li fichye a
        df = pd.read_csv('iot_equipment_monitoring_dataset.csv')
        
        # Tcheke si fichye a vid
        if df.empty:
            st.warning("📁 Fichier CSV trouvé mais vide. Génération de données simulées...")
            return generate_sample_data()
        
        # Tcheke si gen kolòn nan fichye a
        if df.shape[1] == 0:
            st.warning("📁 Fichier CSV sans colonnes. Génération de données simulées...")
            return generate_sample_data()
            
        st.success(f"✅ Fichier CSV chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df
        
    except pd.errors.EmptyDataError:
        st.warning("📁 Fichier CSV vide. Génération de données simulées...")
        return generate_sample_data()
    except Exception as e:
        st.warning(f"📁 Erreur de lecture du CSV ({e}). Génération de données simulées...")
        return generate_sample_data()

# ... [Rete nan lòt fonksyon yo menm jan] ...

@st.cache_resource
def load_model_and_data():
    """Charge les données et entraîne le modèle avec meilleure gestion d'erreur"""
    st.info("🎯 Chargement des données et entraînement du modèle...")
    
    # Premye, kreye yon CSV egzanp si li pa egziste
    create_sample_csv()
    
    # Lè sa a, chaje done yo
    df_raw = load_csv_data()
    df_clean = df_raw.copy()
    
    # Nettoyage
    if 'Fault_Type' in df_clean.columns:
        df_clean['Fault_Type'] = df_clean['Fault_Type'].fillna('None')
    else:
        df_clean['Fault_Type'] = 'None'
    
    # Feature engineering
    if 'Timestamp' in df_clean.columns:
        df_clean['Timestamp'] = pd.to_datetime(df_clean['Timestamp'])
        df_clean['hour'] = df_clean['Timestamp'].dt.hour
    else:
        df_clean['hour'] = 12
    
    df_clean['thermal_stress'] = df_clean['Temperature'] * df_clean['Current'] / 1000
    df_clean['mechanical_stress'] = df_clean['Vibration'] * df_clean['Pressure'] / 1000
    df_clean['power_consumption'] = df_clean['Voltage'] * df_clean['Current']
    
    # Normalisation
    for col in ['Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current']:
        if col in df_clean.columns:
            normalized_col = f'Normalized_{col}'
            df_clean[normalized_col] = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
    
    # Features finales
    base_features = ['Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current']
    optional_features = ['FFT_Feature1', 'FFT_Feature2', 'Anomaly_Score']
    computed_features = [
        'Normalized_Temperature', 'Normalized_Vibration', 'Normalized_Pressure',
        'Normalized_Voltage', 'Normalized_Current', 'hour',
        'thermal_stress', 'mechanical_stress', 'power_consumption'
    ]
    
    features = []
    for feature_list in [base_features, optional_features, computed_features]:
        for feature in feature_list:
            if feature in df_clean.columns:
                features.append(feature)
    
    # Compléter les features manquantes
    for feature in features:
        if feature not in df_clean.columns:
            df_clean[feature] = 0.0
    
    X = df_clean[features]
    y = df_clean['Fault_Type']

    # Modèle ensemble
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(random_state=42)
    svm_model = SVC(probability=True, random_state=42)
    
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf_model), ('xgb', xgb_model), ('svm', svm_model)],
        voting='soft'
    )

    ensemble_model.fit(X, y)
    alert_system = IntelligentAlertSystem(ensemble_model, features)
    
    st.success("✅ Modèle entraîné avec succès !")
    return alert_system, df_clean, ensemble_model, features, X, y

# ... [Rete nan lòt kòd la menm jan] ...
