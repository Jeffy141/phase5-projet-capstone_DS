import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Système de Maintenance Prédictive IoT",
    page_icon="🏭",
    layout="wide"
)

# --- CLASSE SYSTÈME D'ALERTE INTELLIGENT ---
class IntelligentAlertSystem:
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def predict_with_confidence(self, sensor_data):
        """Prédiction avec niveau de confiance"""
        # S'assurer que toutes les features sont présentes
        input_dict = {}
        for feature in self.features:
            if feature in sensor_data:
                input_dict[feature] = sensor_data[feature]
            else:
                # Valeur par défaut pour les features manquantes
                input_dict[feature] = 0.0
        
        input_data = pd.DataFrame([input_dict])[self.features]
        prediction = self.model.predict(input_data)[0]
        probabilities = self.model.predict_proba(input_data)[0]
        confidence = np.max(probabilities)
        class_names = self.model.classes_
        proba_dict = dict(zip(class_names, probabilities))
        return prediction, confidence, proba_dict

    def evaluate_risk(self, sensor_id, prediction, confidence, sensor_data):
        """Évaluation du niveau de risque"""
        risk_levels = {
            'HIGH': {'threshold': 0.85, 'action': 'MAINTENANCE IMMÉDIATE'},
            'MEDIUM': {'threshold': 0.70, 'action': 'SURVEILLANCE RENFORCÉE'},
            'LOW': {'threshold': 0.50, 'action': 'MONITORING STANDARD'}
        }
        
        if prediction != 'None' and confidence > risk_levels['HIGH']['threshold']:
            risk_level = 'HIGH'
        elif prediction != 'None' and confidence > risk_levels['MEDIUM']['threshold']:
            risk_level = 'MEDIUM'
        elif prediction != 'None':
            risk_level = 'LOW'
        else:
            risk_level = 'SAFE'
        
        recommendations = self._generate_recommendations(prediction, sensor_data)
        
        return {
            'sensor_id': sensor_id,
            'prediction': prediction,
            'confidence': confidence,
            'risk_level': risk_level,
            'action': risk_levels.get(risk_level, {}).get('action', 'AUCUNE'),
            'recommendations': recommendations
        }

    def _generate_recommendations(self, prediction, sensor_data):
        """Génération de recommandations personnalisées"""
        recommendations = {
            'None': ["✅ Système opérationnel", "📅 Maintenance planifiée normale"],
            'Electrical Fault': [
                "🔌 Vérifier l'alimentation électrique",
                "🔎 Inspecter les connexions",
                "⚡ Contrôler la stabilité de tension"
            ],
            'Mechanical Failure': [
                "🛑 Arrêt immédiat recommandé",
                "🔧 Inspection mécanique complète",
                "📊 Vérifier l'usure des composants"
            ],
            'Overheating': [
                "🌡️ Réduire la charge immédiatement",
                "❄️ Vérifier le système de refroidissement",
                "🔥 Température critique détectée"
            ]
        }
        return recommendations.get(prediction, ["🔍 Diagnostic à approfondir"])

# --- GÉNÉRATION DE DONNÉES SIMULÉES CORRIGÉE ---
def generate_sample_data():
    """Génère des données IoT simulées complètes"""
    np.random.seed(42)
    n_samples = 1000
    
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
    
    # Génération des défauts
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

# --- CHARGEMENT DU MODÈLE CORRIGÉ ---
@st.cache_resource
def load_model_and_data():  # CORRECTION: "and" au lieu de "amd"
    """Charge et entraîne le modèle avec mise en cache"""
    st.info("🎯 Génération des données et entraînement du modèle...")
    
    # Génération des données
    df_clean = generate_sample_data()
    
    # Feature engineering
    df_clean['Timestamp'] = pd.to_datetime(df_clean['Timestamp'])
    df_clean['hour'] = df_clean['Timestamp'].dt.hour
    
    # Features composites
    df_clean['thermal_stress'] = df_clean['Temperature'] * df_clean['Current'] / 1000
    df_clean['mechanical_stress'] = df_clean['Vibration'] * df_clean['Pressure'] / 1000
    df_clean['power_consumption'] = df_clean['Voltage'] * df_clean['Current']
    
    # Normalisation des features (CORRIGÉ)
    for col in ['Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current']:
        df_clean[f'Normalized_{col}'] = (
            df_clean[col] - df_clean[col].mean()
        ) / df_clean[col].std()

    # Features finales CORRIGÉES (noms exacts des colonnes)
    features = [
        'Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current',
        'FFT_Feature1', 'FFT_Feature2', 'Anomaly_Score',
        'Normalized_Temperature', 'Normalized_Vibration', 'Normalized_Pressure',  # CORRIGÉ
        'Normalized_Voltage', 'Normalized_Current', 'hour',  # CORRIGÉ
        'thermal_stress', 'mechanical_stress', 'power_consumption'
    ]
    
    # Vérification que toutes les features existent
    missing_features = [f for f in features if f not in df_clean.columns]
    if missing_features:
        st.warning(f"Features manquantes : {missing_features}")
        # Créer les features manquantes avec des valeurs par défaut
        for feature in missing_features:
            df_clean[feature] = 0.0
    
    X = df_clean[features]  # CORRIGÉ: utilisation correcte des features
    y = df_clean['Fault_Type']

    # Modèle ensemble
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(random_state=42)
    svm_model = SVC(probability=True, random_state=42)
    
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('svm', svm_model)
        ],
        voting='soft'
    )

    # Entraînement
    ensemble_model.fit(X, y)
    
    # Système d'alerte
    alert_system = IntelligentAlertSystem(ensemble_model, features)
    
    st.success("✅ Modèle entraîné avec succès !")
    return alert_system, df_clean, ensemble_model, features

# --- INTERFACE UTILISATEUR ---
def main():
    st.title("🏭 Système de Maintenance Prédictive IoT")
    st.markdown("**Projet Capstone Data Science - Version Corrigée**")
    
    # Sidebar avec informations
    with st.sidebar:
        st.header("📊 À propos")
        st.markdown("""
        **Technologies utilisées :**
        - Python, Scikit-learn, XGBoost
        - Random Forest, SVM, Modèle Ensemble
        - Streamlit pour l'interface
        
        **Types de défauts détectés :**
        - ✅ Aucun défaut
        - 🔌 Défaut électrique
        - 🔧 Défaut mécanique  
        - 🌡️ Surchauffe
        """)
        
        st.header("🎯 Objectifs")
        st.markdown("""
        - Classification multi-classes
        - Détection temps réel
        - Réduction des coûts de maintenance
        - Alertes intelligentes
        """)
    
    # Chargement du modèle
    try:
        alert_system, df_demo, model, features = load_model_and_data()  # CORRIGÉ
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return
    
    st.header("🔍 Simulation de Prédiction en Temps Réel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Paramètres du Capteur")
        
        # Sélection de scénarios
        scenario = st.selectbox(
            "Choisir un scénario :",
            ["🎯 Normal", "⚠️ Défaut Électrique", "🔧 Défaut Mécanique", "🌡️ Surchauffe"]
        )
        
        # Valeurs par défaut selon le scénario
        if scenario == "🎯 Normal":
            temp, vib, press, volt, curr = 65.0, 3.5, 95.0, 220.0, 15.0
        elif scenario == "⚠️ Défaut Électrique":
            temp, vib, press, volt, curr = 75.0, 4.0, 110.0, 190.0, 22.0
        elif scenario == "🔧 Défaut Mécanique":
            temp, vib, press, volt, curr = 70.0, 9.5, 140.0, 215.0, 16.0
        else:  # Surchauffe
            temp, vib, press, volt, curr = 92.0, 5.0, 105.0, 225.0, 18.0
        
        temperature = st.slider("🌡️ Temperature (°C)", 20.0, 120.0, temp, 0.1)
        vibration = st.slider("📊 Vibration (m/s²)", 0.0, 15.0, vib, 0.1)
        pressure = st.slider("💨 Pressure (kPa)", 50.0, 200.0, press, 0.1)
        voltage = st.slider("⚡ Voltage (V)", 180.0, 250.0, volt, 0.1)
        current = st.slider("🔋 Current (A)", 5.0, 25.0, curr, 0.1)
    
    with col2:
        st.subheader("🎯 Résultats du Diagnostic")
        
        if st.button("🚀 Lancer le Diagnostic", type="primary", use_container_width=True):
            # Préparation des données de capteur
            sensor_data = {
                'Temperature': temperature,
                'Vibration': vibration,
                'Pressure': pressure,
                'Voltage': voltage,
                'Current': current,
                'FFT_Feature1': np.random.uniform(0, 1),
                'FFT_Feature2': np.random.uniform(0, 1),
                'Anomaly_Score': np.random.uniform(0, 1),
                'hour': pd.Timestamp.now().hour,
                'thermal_stress': temperature * current / 1000,
                'mechanical_stress': vibration * pressure / 1000,
                'power_consumption': voltage * current,
                'Normalized_Temperature': 0.0,
                'Normalized_Vibration': 0.0,
                'Normalized_Pressure': 0.0,
                'Normalized_Voltage': 0.0,
                'Normalized_Current': 0.0
            }
            
            # Prédiction
            with st.spinner("🔍 Analyse en cours..."):
                try:
                    prediction, confidence, proba_dict = alert_system.predict_with_confidence(sensor_data)
                    alert_info = alert_system.evaluate_risk("SENSOR_001", prediction, confidence, sensor_data)
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction : {e}")
                    return
            
            # Affichage des résultats
            st.subheader("📋 Diagnostic Complet")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prédiction", alert_info['prediction'])
            with col2:
                risk_color = {
                    'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🔵', 'SAFE': '🟢'
                }
                st.metric("Niveau de Risque", 
                         f"{risk_color.get(alert_info['risk_level'], '⚪')} {alert_info['risk_level']}")
            with col3:
                st.metric("Confiance", f"{alert_info['confidence']:.1%}")
            
            # Alerte colorée
            if alert_info['risk_level'] == 'HIGH':
                st.error(f"🚨 **{alert_info['action']}**")
            elif alert_info['risk_level'] == 'MEDIUM':
                st.warning(f"⚠️ **{alert_info['action']}**")
            elif alert_info['risk_level'] == 'LOW':
                st.info(f"ℹ️ **{alert_info['action']}**")
            else:
                st.success(f"✅ **{alert_info['action']}**")
            
            # Recommandations
            with st.expander("📋 Recommandations détaillées", expanded=True):
                for i, rec in enumerate(alert_info['recommendations'], 1):
                    st.write(f"{i}. {rec}")
            
            # Probabilités
            with st.expander("📊 Probabilités par type de défaut"):
                prob_df = pd.DataFrame(proba_dict, index=['Probabilité']).T
                prob_df_sorted = prob_df.sort_values('Probabilité', ascending=False)
                st.dataframe(prob_df_sorted.style.format("{:.1%}"))

if __name__ == "__main__":
    main()
