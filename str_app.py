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

# --- CLASSE SYSTÈME D'ALERTE INTELLIGENT ---
class IntelligentAlertSystem:
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def predict_with_confidence(self, sensor_data):
        """Prédiction avec niveau de confiance"""
        input_dict = {}
        for feature in self.features:
            if feature in sensor_data:
                input_dict[feature] = sensor_data[feature]
            else:
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

# --- GESTION DES DONNÉES AVEC SUPPORT FRANÇAIS ---
def find_csv_file():
    """Cherche le fichier CSV dans différents noms possibles"""
    import os
    possible_files = [
        'ensemble_de_donnees_de_surveillance_iot.csv',  # Nom français
        'iot_equipment_monitoring_dataset.csv',         # Nom anglais
       
    ]
    
    for file_name in possible_files:
        if os.path.exists(file_name):
            st.info(f"📁 Fichier trouvé: {file_name}")
            return file_name
    
    return None

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

@st.cache_data
def load_csv_data():
    """Charge le fichier CSV avec gestion multi-noms"""
    import os
    
    # Chercher le fichier CSV
    csv_file = find_csv_file()
    
    if csv_file is None:
        st.warning("📁 Aucun fichier CSV trouvé. Génération de données simulées...")
        return generate_sample_data()
    
    try:
        # Essayer de lire le fichier
        st.info(f"🔄 Lecture du fichier: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Vérifier si le fichier est vide
        if df.empty:
            st.warning("📁 Fichier CSV trouvé mais vide. Génération de données simulées...")
            return generate_sample_data()
        
        # Vérifier s'il y a des colonnes
        if df.shape[1] == 0:
            st.warning("📁 Fichier CSV sans colonnes. Génération de données simulées...")
            return generate_sample_data()
            
        st.success(f"✅ Fichier CSV chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        # Afficher les premières colonnes pour debug
        st.info(f"📊 Colonnes détectées: {list(df.columns[:8])}...")
        
        return df
        
    except pd.errors.EmptyDataError:
        st.warning("📁 Fichier CSV vide. Génération de données simulées...")
        return generate_sample_data()
    except Exception as e:
        st.warning(f"📁 Erreur de lecture du CSV ({e}). Génération de données simulées...")
        return generate_sample_data()

# --- CHARGEMENT DU MODÈLE AMÉLIORÉ ---
@st.cache_resource
def load_model_and_data():
    """Charge et entraîne le modèle avec gestion des données réelles"""
    st.info("🎯 Chargement des données et entraînement du modèle...")
    
    # Charger les données (réelles ou simulées)
    df_raw = load_csv_data()
    df_clean = df_raw.copy()
    
    # Mapping des colonnes françaises vers anglaises
    column_mapping = {
        'Capteur_ID': 'Sensor_ID',
        'Température': 'Temperature', 
        'Vibration': 'Vibration',
        'Pression': 'Pressure',
        'Tension': 'Voltage',
        'Courant': 'Current',
        'Type_Defaut': 'Fault_Type',
        'Statut_Defaut': 'Fault_Status',
        'Horodatage': 'Timestamp'
    }
    
    # Renommer les colonnes françaises si elles existent
    for fr_col, en_col in column_mapping.items():
        if fr_col in df_clean.columns and en_col not in df_clean.columns:
            df_clean[en_col] = df_clean[fr_col]
            st.info(f"🔄 Colonne renommée: {fr_col} -> {en_col}")
    
    # Vérifier et créer les colonnes essentielles si manquantes
    essential_columns = ['Sensor_ID', 'Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current', 'Fault_Type']
    for col in essential_columns:
        if col not in df_clean.columns:
            if col == 'Sensor_ID':
                df_clean[col] = [f'SENSOR_{i:03d}' for i in range(len(df_clean))]
            elif col == 'Fault_Type':
                df_clean[col] = 'None'
            else:
                df_clean[col] = 0.0
            st.warning(f"⚠️ Colonne '{col}' manquante, création avec valeurs par défaut")
    
    # Feature engineering
    if 'Timestamp' in df_clean.columns:
        df_clean['Timestamp'] = pd.to_datetime(df_clean['Timestamp'])
        df_clean['hour'] = df_clean['Timestamp'].dt.hour
    else:
        df_clean['hour'] = 12
    
    # Features composites
    df_clean['thermal_stress'] = df_clean['Temperature'] * df_clean['Current'] / 1000
    df_clean['mechanical_stress'] = df_clean['Vibration'] * df_clean['Pressure'] / 1000
    df_clean['power_consumption'] = df_clean['Voltage'] * df_clean['Current']
    
    # Normalisation des features
    for col in ['Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current']:
        if col in df_clean.columns:
            df_clean[f'Normalized_{col}'] = (
                df_clean[col] - df_clean[col].mean()
            ) / df_clean[col].std()
        else:
            df_clean[f'Normalized_{col}'] = 0.0

    # Features finales
    features = [
        'Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current',
        'FFT_Feature1', 'FFT_Feature2', 'Anomaly_Score',
        'Normalized_Temperature', 'Normalized_Vibration', 'Normalized_Pressure',
        'Normalized_Voltage', 'Normalized_Current', 'hour',
        'thermal_stress', 'mechanical_stress', 'power_consumption'
    ]
    
    # Vérification que toutes les features existent
    missing_features = [f for f in features if f not in df_clean.columns]
    if missing_features:
        st.warning(f"Features manquantes créées: {missing_features}")
        for feature in missing_features:
            df_clean[feature] = 0.0
    
    X = df_clean[features]
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

    # Entraînement avec progression
    with st.spinner("🤖 Entraînement du modèle en cours..."):
        ensemble_model.fit(X, y)
    
    # Système d'alerte
    alert_system = IntelligentAlertSystem(ensemble_model, features)
    
    st.success("✅ Modèle entraîné avec succès !")
    return alert_system, df_clean, ensemble_model, features, X, y

# --- FONCTIONS DE VISUALISATION ---
def create_confusion_matrix(model, X, y):
    """Crée la matrice de confusion"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    ax.set_title('Matrice de Confusion', fontsize=14, fontweight='bold')
    ax.set_xlabel('Prédictions')
    ax.set_ylabel('Valeurs Réelles')
    return fig

def create_feature_importance(model, features):
    """Crée le graphique d'importance des features"""
    if hasattr(model.estimators_[0], 'feature_importances_'):
        importances = model.estimators_[0].feature_importances_
    else:
        importances = np.ones(len(features)) / len(features)
    
    feature_imp = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_imp.head(10), y='feature', x='importance', ax=ax)
    ax.set_title('Top 10 - Importance des Variables', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    return fig, feature_imp

def create_fault_distribution(df):
    """Crée la distribution des défauts"""
    fault_counts = df['Fault_Type'].value_counts()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Camembert
    colors = ['lightgreen', 'red', 'orange', 'yellow']
    ax1.pie(fault_counts.values, labels=fault_counts.index, autopct='%1.1f%%', 
            colors=colors[:len(fault_counts)], startangle=90)
    ax1.set_title('Distribution des Types de Défauts')
    
    # Barres
    sns.barplot(x=fault_counts.index, y=fault_counts.values, ax=ax2, palette=colors[:len(fault_counts)])
    ax2.set_title('Nombre de Défauts par Type')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

# --- INTERFACE UTILISATEUR AMÉLIORÉE ---
def main():
    st.title("🏭 Système de Maintenance Prédictive IoT")
    st.markdown("**Projet Capstone - Présentation Complète avec Visualisations**")
    
    # Barre de progression
    with st.expander("🎯 Statut du Chargement", expanded=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Chargement du modèle
        try:
            status_text.text("🔍 Recherche des données...")
            progress_bar.progress(25)
            
            alert_system, df_clean, model, features, X, y = load_model_and_data()
            
            status_text.text("✅ Données chargées avec succès!")
            progress_bar.progress(100)
            
        except Exception as e:
            st.error(f"Erreur lors du chargement : {e}")
            return
    
    # Supprimer la barre de progression après chargement
    progress_bar.empty()
    status_text.empty()
    
    # Onglets pour une meilleure organisation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Simulation Temps Réel", 
        "📊 Analyse des Données", 
        "🤖 Performance du Modèle",
        "📈 Visualisations Présentation"
    ])
    
    with tab1:
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
                sensor_data = {}
                for feature in features:
                    if feature == 'Temperature': 
                        sensor_data[feature] = temperature
                    elif feature == 'Vibration': 
                        sensor_data[feature] = vibration
                    elif feature == 'Pressure': 
                        sensor_data[feature] = pressure
                    elif feature == 'Voltage': 
                        sensor_data[feature] = voltage
                    elif feature == 'Current': 
                        sensor_data[feature] = current
                    elif feature == 'thermal_stress': 
                        sensor_data[feature] = temperature * current / 1000
                    elif feature == 'mechanical_stress': 
                        sensor_data[feature] = vibration * pressure / 1000
                    elif feature == 'power_consumption': 
                        sensor_data[feature] = voltage * current
                    elif feature == 'hour': 
                        sensor_data[feature] = pd.Timestamp.now().hour
                    else: 
                        sensor_data[feature] = np.random.uniform(0, 1)
                
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
                    st.dataframe(prob_df_sorted.style.format("{:.1%}").background_gradient(cmap='RdYlGn_r'))
    
    with tab2:
        st.header("📊 Analyse Exploratoire des Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution des Défauts")
            fig_dist = create_fault_distribution(df_clean)
            st.pyplot(fig_dist)
        
        with col2:
            st.subheader("Statistiques Descriptives")
            numeric_cols = ['Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current']
            numeric_cols = [col for col in numeric_cols if col in df_clean.columns]
            st.dataframe(df_clean[numeric_cols].describe().style.background_gradient(cmap='Blues'))
            
            st.subheader("Échantillon des Données")
            st.dataframe(df_clean.head(10))
    
    with tab3:
        st.header("🤖 Performance du Modèle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Matrice de Confusion")
            fig_cm = create_confusion_matrix(model, X, y)
            st.pyplot(fig_cm)
        
        with col2:
            st.subheader("Importance des Features")
            fig_imp, feature_imp = create_feature_importance(model, features)
            st.pyplot(fig_imp)
            
            st.subheader("Top 10 Features")
            st.dataframe(feature_imp.head(10).style.background_gradient(cmap='Greens'))
    
    with tab4:
        st.header("📈 Visualisations pour Présentation")
        
        st.info("🎯 **Graphiques prêts pour votre présentation Capstone**")
        
        # Graphique 1: Distribution des défauts
        st.subheader("1. Distribution des Types de Défauts")
        fig1 = create_fault_distribution(df_clean)
        st.pyplot(fig1)
        
        # Graphique 2: Matrice de confusion
        st.subheader("2. Performance du Modèle - Matrice de Confusion")
        fig2 = create_confusion_matrix(model, X, y)
        st.pyplot(fig2)
        
        # Graphique 3: Importance des features
        st.subheader("3. Importance des Variables Prédictives")
        fig3, _ = create_feature_importance(model, features)
        st.pyplot(fig3)

# Sidebar avec informations
with st.sidebar:
    st.header("📊 À propos")
    st.markdown("""
    **Fichiers CSV supportés:**
    - `ensemble_de_donnees_de_surveillance_iot.csv` 🇫🇷
    - `iot_equipment_monitoring_dataset.csv` 🇬🇧
    - Autres noms: `donnees_iot.csv`, `data_iot.csv`
    
    **Technologies utilisées :**
    - Python, Scikit-learn, XGBoost
    - Random Forest, SVM, Modèle Ensemble
    - Streamlit pour l'interface
    """)
    
    st.header("🎯 Points Clés Présentation")
    st.markdown("""
    1. **Simulation temps réel**
    2. **Performance du modèle** 
    3. **Analyse des données**
    4. **Visualisations professionnelles**
    5. **Recommandations business**
    """)
    
    # Information sur les données chargées
    try:
        if 'df_clean' in locals():
            st.header("📈 Métriques")
            st.metric("Nombre d'échantillons", len(df_clean))
            st.metric("Types de défauts", df_clean['Fault_Type'].nunique())
            fault_rate = (df_clean['Fault_Type'] != 'None').mean() * 100
            st.metric("Taux de défaut", f"{fault_rate:.1f}%")
    except:
        pass

if __name__ == "__main__":
    main()
