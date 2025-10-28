import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split # Assurez-vous que cette ligne est bien là!
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Système de Maintenance Prédictive IoT",
    page_icon="🏭",
    layout="wide"
)

# Style pour les graphiques (on enlève 'plt.style.use')
sns.set_palette("husl") # Gardez ceci pour les couleurs de seaborn

# --- CLASSE SYSTÈME D'ALERTE INTELLIGENT ---
# ... le reste du code ...

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

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_csv_data():
    """Charge le fichier CSV réel"""
    try:
        df = pd.read_csv('iot_equipment_monitoring_dataset.csv')
        st.success(f"✅ Fichier CSV chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df
    except FileNotFoundError:
        st.warning("📁 Fichier CSV non trouvé. Génération de données simulées...")
        return generate_sample_data()

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

# --- CHARGEMENT ET PRÉPARATION DES DONNÉES ---
@st.cache_resource
def load_model_and_data():
    """Charge les données et entraîne le modèle"""
    st.info("🎯 Chargement des données et entraînement du modèle...")
    
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
    
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf_model), ('xgb', xgb_model)],
        voting='soft'
    )

    # 🟢 ÉTAPE CRUCIALE : Entraîner le modèle !
    ensemble_model.fit(X, y) 
    
    # Initialiser le système d'alerte
    alert_system = IntelligentAlertSystem(ensemble_model, features)
    
    st.success("🤖 Modèle entraîné et prêt à l'emploi !")
    
    # 🟢 RETOURNER TOUTES LES VARIABLES DANS L'ORDRE ATTENDU
    return alert_system, df_clean, ensemble_model, features, X, y
    
# --- FONCTIONS DE VISUALISATION ---
def create_confusion_matrix(model, X, y):
    """Crée la matrice de confusion"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    ax.set_title('MATRICE DE CONFUSION', fontsize=16, fontweight='bold')
    ax.set_xlabel('Prédictions')
    ax.set_ylabel('Valeurs Réelles')
    
    # Ajouter l'accuracy
    accuracy = accuracy_score(y_test, y_pred)
    ax.text(0.5, -0.1, f'Accuracy: {accuracy:.3f}', transform=ax.transAxes, 
            ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    return fig

def create_feature_importance(model, features):
    """Crée le graphique d'importance des features"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # Pour VotingClassifier, prendre la moyenne des importances
        importances = np.mean([est.feature_importances_ for est in model.estimators_ if hasattr(est, 'feature_importances_')], axis=0)
    
    feature_imp = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=feature_imp.head(15), y='feature', x='importance', ax=ax)
    ax.set_title('TOP 15 - IMPORTANCE DES VARIABLES', fontsize=16, fontweight='bold')
    ax.set_xlabel('Importance')
    return fig, feature_imp

def create_fault_distribution(df):
    """Crée la distribution des défauts"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Camembert
    fault_dist = df['Fault_Type'].value_counts()
    colors = ['lightgreen', 'red', 'orange', 'yellow']
    ax1.pie(fault_dist.values, labels=fault_dist.index, autopct='%1.1f%%', 
            startangle=90, colors=colors[:len(fault_dist)])
    ax1.set_title('Distribution des Types de Défauts', fontweight='bold')
    
    # Barres
    sns.barplot(x=fault_dist.index, y=fault_dist.values, ax=ax2, palette=colors[:len(fault_dist)])
    ax2.set_title('Nombre de Défauts par Type', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def create_sensor_risk_analysis(df):
    """Analyse des capteurs à risque"""
    sensor_analysis = df.groupby('Sensor_ID').agg({
        'Fault_Status': ['count', 'sum', 'mean'], 
        'Temperature': 'max', 
        'Vibration': 'max'
    }).round(3)
    
    sensor_analysis.columns = ['total_mesures', 'defauts_total', 'taux_defaut', 'temp_max', 'vibration_max']
    high_risk_sensors = sensor_analysis[sensor_analysis['taux_defaut'] > 0.1].sort_values('taux_defaut', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Top 10 capteurs à risque
    top_10 = high_risk_sensors['taux_defaut'].head(10)
    colors = ['red' if x > 0.3 else 'orange' for x in top_10.values]
    top_10.plot(kind='bar', ax=ax1, color=colors)
    ax1.set_title('TOP 10 - TAUX DE DÉFAUTS PAR CAPTEUR', fontweight='bold')
    ax1.set_ylabel('Taux de Défauts')
    ax1.tick_params(axis='x', rotation=45)
    
    # Scatter plot température vs vibration
    scatter = sns.scatterplot(data=sensor_analysis, x='temp_max', y='vibration_max',
                   size='taux_defaut', hue='taux_defaut', sizes=(20, 200), ax=ax2, palette='viridis')
    ax2.set_title('PROFIL DES CAPTEURS - TEMPÉRATURE vs VIBRATION', fontweight='bold')
    ax2.set_xlabel('Température Max (°C)')
    ax2.set_ylabel('Vibration Max (m/s²)')
    
    plt.tight_layout()
    return fig, high_risk_sensors.head(10)

def create_correlation_heatmap(df):
    """Crée la heatmap de corrélation"""
    numeric_features = ['Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current', 
                       'FFT_Feature1', 'FFT_Feature2', 'Anomaly_Score', 'Fault_Status']
    numeric_features = [f for f in numeric_features if f in df.columns]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation_matrix = df[numeric_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, fmt='.2f', ax=ax, cbar_kws={"shrink": .8})
    ax.set_title('MATRICE DE CORRÉLATION - VARIABLES IoT', fontsize=16, fontweight='bold')
    return fig

def create_sensor_visualization(sensor_data, prediction):
    """Crée la visualisation des données du capteur"""
    fig = plt.figure(figsize=(15, 10))
    
    # Graphique radar pour les paramètres principaux
    ax1 = fig.add_subplot(221, polar=True)
    parameters = ['Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current']
    values = [sensor_data.get(p, 0) for p in parameters]
    
    # Normalisation pour le radar
    max_vals = [120, 15, 200, 250, 25]  # Valeurs maximales typiques
    normalized_vals = [v/max_vals[i] for i, v in enumerate(values)]
    
    angles = np.linspace(0, 2*np.pi, len(parameters), endpoint=False).tolist()
    angles += angles[:1]
    normalized_vals += normalized_vals[:1]
    
    ax1.plot(angles, normalized_vals, 'o-', linewidth=2, label='Capteur')
    ax1.fill(angles, normalized_vals, alpha=0.25)
    ax1.set_thetagrids(np.degrees(angles[:-1]), parameters)
    ax1.set_title('PROFIL DU CAPTEUR (Radar)', size=14, fontweight='bold')
    ax1.grid(True)
    
    # Barres des valeurs actuelles
    ax2 = fig.add_subplot(222)
    colors = ['red' if v > max_vals[i]*0.8 else 'green' for i, v in enumerate(values)]
    bars = ax2.bar(parameters, values, color=colors, alpha=0.7)
    ax2.set_title('VALEURS DES CAPTEURS', fontweight='bold')
    ax2.set_ylabel('Valeurs')
    ax2.tick_params(axis='x', rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.1f}', ha='center', va='bottom')
    
    # Seuils de danger
    danger_thresholds = [85, 8, 130, 210, 20]
    for i, (param, danger) in enumerate(zip(parameters, danger_thresholds)):
        ax2.axhline(y=danger, color='red', linestyle='--', alpha=0.5)
        ax2.text(len(parameters)-0.5, danger + (i*2), f'Seuil {param}: {danger}', 
                fontsize=8, color='red', ha='right')
    
    # Distribution temporelle simulée
    ax3 = fig.add_subplot(223)
    time_points = np.arange(24)
    temp_trend = sensor_data['Temperature'] + np.random.normal(0, 5, 24)
    vib_trend = sensor_data['Vibration'] + np.random.normal(0, 1, 24)
    
    ax3.plot(time_points, temp_trend, label='Température', color='red', linewidth=2)
    ax3.set_ylabel('Température (°C)', color='red')
    ax3.tick_params(axis='y', labelcolor='red')
    ax3.set_xlabel('Heures')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(time_points, vib_trend, label='Vibration', color='blue', linewidth=2)
    ax3_twin.set_ylabel('Vibration (m/s²)', color='blue')
    ax3_twin.tick_params(axis='y', labelcolor='blue')
    
    ax3.set_title('TENDANCE TEMPORELLE (24h)', fontweight='bold')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    
    # Statut de prédiction
    ax4 = fig.add_subplot(224)
    status_colors = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'yellow', 'SAFE': 'green'}
    status = 'HIGH' if prediction != 'None' else 'SAFE'
    
    ax4.text(0.5, 0.6, f'STATUT: {prediction}', ha='center', va='center', 
             fontsize=20, fontweight='bold', color=status_colors.get(status, 'black'))
    ax4.text(0.5, 0.3, f'NIVEAU: {status}', ha='center', va='center', 
             fontsize=16, color=status_colors.get(status, 'black'))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('DIAGNOSTIC ACTUEL', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_performance_dashboard(model, X, y):
    """Crée un dashboard de performance complet"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    
    # Métriques de performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy par classe
    classes = model.classes_
    class_accuracy = []
    for cls in classes:
        if cls in report:
            class_accuracy.append(report[cls]['precision'])
    
    ax1.bar(classes, class_accuracy, color=['lightblue', 'lightcoral', 'lightgreen', 'yellow'])
    ax1.set_title('PRÉCISION PAR CLASSE', fontweight='bold')
    ax1.set_ylabel('Précision')
    ax1.tick_params(axis='x', rotation=45)
    
    # Distribution des probabilités
    y_proba = model.predict_proba(X_test)
    proba_df = pd.DataFrame(y_proba, columns=classes)
    proba_df['true_class'] = y_test.reset_index(drop=True)
    
    for cls in classes:
        cls_proba = proba_df[proba_df['true_class'] == cls][cls]
        ax2.hist(cls_proba, alpha=0.6, label=cls, bins=20)
    
    ax2.set_title('DISTRIBUTION DES PROBABILITÉS', fontweight='bold')
    ax2.set_xlabel('Probabilité')
    ax2.set_ylabel('Fréquence')
    ax2.legend()
    
    # Learning curve (simulée)
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    test_scores = []
    
    for size in train_sizes:
        n_train = int(size * len(X_train))
        model.fit(X_train[:n_train], y_train[:n_train])
        train_scores.append(accuracy_score(y_train[:n_train], model.predict(X_train[:n_train])))
        test_scores.append(accuracy_score(y_test, model.predict(X_test)))
    
    ax3.plot(train_sizes, train_scores, 'o-', label='Train', color='blue')
    ax3.plot(train_sizes, test_scores, 'o-', label='Test', color='red')
    ax3.set_title('COURBE D\'APPRENTISSAGE', fontweight='bold')
    ax3.set_xlabel('Taille de l\'ensemble d\'entraînement')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Métriques globales
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        accuracy,
        report['weighted avg']['precision'],
        report['weighted avg']['recall'],
        report['weighted avg']['f1-score']
    ]
    
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold']
    bars = ax4.bar(metrics, values, color=colors)
    ax4.set_title('MÉTRIQUES GLOBALES DU MODÈLE', fontweight='bold')
    ax4.set_ylabel('Score')
    ax4.set_ylim(0, 1)
    
    # Ajouter les valeurs sur les barres
    for bar, value in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

# --- INTERFACE PRINCIPALE ---
def main():
    st.title("🏭 Système de Maintenance Prédictive IoT")
    st.markdown("**Projet Capstone - Présentation Complète avec Visualisations**")
    
    # Barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Chargement des données
    try:
        status_text.text("🎯 Chargement des données...")
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
    
    # Onglets pour l'organisation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Simulation", 
        "📊 Analyse Données", 
        "🤖 Performance Modèle",
        "📈 Graphiques Présentation",
        "🏆 Top 10 Capteurs",
        "📋 Rapport Complet"
    ])
    
    with tab1:
        st.header("🔍 Simulation en Temps Réel")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Paramètres du Capteur")
            
            # Contrôles avancés
            temperature = st.slider("🌡️ Temperature (°C)", 20.0, 120.0, 65.0, 0.1)
            vibration = st.slider("📊 Vibration (m/s²)", 0.0, 15.0, 3.5, 0.1)
            pressure = st.slider("💨 Pressure (kPa)", 50.0, 200.0, 95.0, 0.1)
            voltage = st.slider("⚡ Voltage (V)", 180.0, 250.0, 220.0, 0.1)
            current = st.slider("🔋 Current (A)", 5.0, 25.0, 15.0, 0.1)
            
            # Exemples prédéfinis pour démo
            scenario = st.selectbox("Scénario de démonstration :", 
                                  ["Normal", "Défaut Électrique", "Défaut Mécanique", "Surchauffe"])
            
            if scenario == "Défaut Électrique":
                temperature, vibration, pressure, voltage, current = 75.0, 4.0, 110.0, 190.0, 22.0
            elif scenario == "Défaut Mécanique":
                temperature, vibration, pressure, voltage, current = 70.0, 9.5, 140.0, 215.0, 16.0
            elif scenario == "Surchauffe":
                temperature, vibration, pressure, voltage, current = 92.0, 5.0, 105.0, 225.0, 18.0
        
        with col2:
            st.subheader("🎯 Résultats du Diagnostic")
            
            if st.button("🚀 Lancer le Diagnostic", type="primary", use_container_width=True):
                # Préparation des données
                sensor_data = {}
                for feature in features:
                    if feature == 'Temperature': sensor_data[feature] = temperature
                    elif feature == 'Vibration': sensor_data[feature] = vibration
                    elif feature == 'Pressure': sensor_data[feature] = pressure
                    elif feature == 'Voltage': sensor_data[feature] = voltage
                    elif feature == 'Current': sensor_data[feature] = current
                    elif feature == 'thermal_stress': sensor_data[feature] = temperature * current / 1000
                    elif feature == 'mechanical_stress': sensor_data[feature] = vibration * pressure / 1000
                    elif feature == 'power_consumption': sensor_data[feature] = voltage * current
                    elif feature == 'hour': sensor_data[feature] = pd.Timestamp.now().hour
                    else: sensor_data[feature] = 0.0
                
                # Prédiction
                with st.spinner("🔍 Analyse en cours..."):
                    prediction, confidence, proba_dict = alert_system.predict_with_confidence(sensor_data)
                    alert_info = alert_system.evaluate_risk("SENSOR_001", prediction, confidence, sensor_data)
                
                # Résultats
                col1, col2, col3 = st.columns(3)
                
                risk_color = {
                    'HIGH': 'red', 
                    'MEDIUM': 'orange', 
                    'LOW': 'yellow', 
                    'SAFE': 'green'
                }
                
                col1.metric("Prédiction", alert_info['prediction'])
                col2.metric("Niveau de Risque", alert_info['risk_level'])
                col3.metric("Confiance", f"{alert_info['confidence']:.1%}")
                
                # Alerte
                if alert_info['risk_level'] == 'HIGH':
                    st.error(f"🚨 **{alert_info['action']}**")
                elif alert_info['risk_level'] == 'MEDIUM':
                    st.warning(f"⚠️ **{alert_info['action']}**")
                elif alert_info['risk_level'] == 'LOW':
                    st.info(f"ℹ️ **{alert_info['action']}**")
                else:
                    st.success(f"✅ **{alert_info['action']}**")
                
                # Graphique de visualisation du capteur
                st.subheader("📈 Visualisation des Données du Capteur")
                fig_sensor = create_sensor_visualization(sensor_data, prediction)
                st.pyplot(fig_sensor)
                
                # Recommandations et probabilités
                col_rec, col_prob = st.columns(2)
                
                with col_rec:
                    with st.expander("📋 Recommandations détaillées", expanded=True):
                        for i, rec in enumerate(alert_info['recommendations'], 1):
                            st.write(f"{i}. {rec}")
                
                with col_prob:
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
            st.subheader("Matrice de Corrélation")
            fig_corr = create_correlation_heatmap(df_clean)
            st.pyplot(fig_corr)
        
        # Statistiques descriptives
        st.subheader("📋 Statistiques Descriptives")
        numeric_cols = ['Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current']
        numeric_cols = [col for col in numeric_cols if col in df_clean.columns]
        st.dataframe(df_clean[numeric_cols].describe().style.background_gradient(cmap='Blues'))
    
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
        
        # Dashboard de performance
        st.subheader("📊 Dashboard de Performance Complet")
        fig_perf = create_performance_dashboard(model, X, y)
        st.pyplot(fig_perf)
    
    with tab4:
        st.header("📈 Graphiques pour Présentation")
        
        st.info("🎯 **Graphiques prêts pour votre présentation**")
        
        # Graphique 1: Distribution des défauts
        st.subheader("1. Distribution des Types de Défauts")
        fig1 = create_fault_distribution(df_clean)
        st.pyplot(fig1)
        
        # Graphique 2: Matrice de confusion
        st.subheader("2. Matrice de Confusion du Modèle")
        fig2 = create_confusion_matrix(model, X, y)
        st.pyplot(fig2)
        
        # Graphique 3: Importance des features
        st.subheader("3. Importance des Variables")
        fig3, _ = create_feature_importance(model, features)
        st.pyplot(fig3)
        
        # Graphique 4: Corrélations
        st.subheader("4. Analyse des Corrélations")
        fig4 = create_correlation_heatmap(df_clean)
        st.pyplot(fig4)
    
    with tab5:
        st.header("🏆 Top 10 Capteurs à Risque")
        
        fig_risk, top_sensors = create_sensor_risk_analysis(df_clean)
        st.pyplot(fig_risk)
        
        st.subheader("📋 Liste Détaillée des Capteurs à Risque")
        st.dataframe(top_sensors.style.background_gradient(cmap='Reds', subset=['taux_defaut']))
        
        # Explication des résultats
        st.subheader("📝 Analyse des Résultats")
        st.markdown("""
        **Interprétation :**
        - 🔴 **Capteurs à haut risque** : Taux de défaut > 10%
        - 🟡 **Risque moyen** : Taux de défaut 5-10%  
        - 🟢 **Faible risque** : Taux de défaut < 5%
        
        **Recommandations :**
        - Prioriser la maintenance sur les capteurs en rouge
        - Surveiller les tendances des capteurs en orange
        - Maintenir la surveillance standard pour les verts
        """)
    
    with tab6:
        st.header("📋 Rapport Complet du Système")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Métriques Clés")
            
            # Calcul des métriques
            total_sensors = df_clean['Sensor_ID'].nunique()
            fault_rate = df_clean['Fault_Status'].mean() * 100
            avg_temperature = df_clean['Temperature'].mean()
            avg_vibration = df_clean['Vibration'].mean()
            
            st.metric("Nombre total de capteurs", total_sensors)
            st.metric("Taux de défaut global", f"{fault_rate:.1f}%")
            st.metric("Température moyenne", f"{avg_temperature:.1f}°C")
            st.metric("Vibration moyenne", f"{avg_vibration:.1f} m/s²")
        
        with col2:
            st.subheader("🎯 Performance du Modèle")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.metric("Accuracy globale", f"{accuracy:.3f}")
            st.metric("Nombre de features", len(features))
            st.metric("Classes prédites", len(model.classes_))
        
        # Résumé exécutif
        st.subheader("📋 Résumé Exécutif")
        st.markdown(f"""
        **Système de Maintenance Prédictive IoT - Rapport de Performance**
        
        - **📊 Couverture des données** : {len(df_clean)} échantillons analysés
        - **🤖 Performance du modèle** : {accuracy:.1%} de précision
        - **🔧 Types de défauts détectés** : {len(model.classes_)}
        - **🎯 Capteurs à risque identifiés** : {len(top_sensors) if 'top_sensors' in locals() else 'N/A'}
        - **💡 Recommandations** : Système opérationnel et prêt pour le déploiement
        
        **Points forts :**
        ✅ Détection précoce des défauts  
        ✅ Interface intuitive  
        ✅ Modèle ensemble robuste  
        ✅ Visualisations complètes  
        """)

# Sidebar avec informations
with st.sidebar:
    st.header("📊 À propos")
    st.markdown("""
    **Technologies utilisées :**
    - Python, Scikit-learn, XGBoost
    - Random Forest, SVM, Modèle Ensemble
    - Streamlit pour l'interface
    
    **Graphiques inclus :**
    - 📈 Matrice de confusion
    - 📊 Importance des features
    - 🎯 Distribution des défauts
    - 🔗 Corrélations
    - 🏆 Top 10 capteurs à risque
    - 📉 Visualisation temps réel
    """)
    
    st.header("🎯 Pour la Présentation")
    st.markdown("""
    **Points clés à montrer :**
    1. Simulation temps réel
    2. Performance du modèle
    3. Analyse des données
    4. Graphiques explicatifs
    5. Recommandations business
    """)
    
    # Ajouter un sélecteur de thème
    st.header("🎨 Personnalisation")
    theme = st.selectbox("Choisir le thème des graphiques:", 
                        ["Default", "Dark", "Pastel", "Bright"])
    
    if theme == "Dark":
        plt.style.use('dark_background')
        sns.set_palette("husl")
    elif theme == "Pastel":
        sns.set_palette("pastel")
    elif theme == "Bright":
        sns.set_palette("bright")

if __name__ == "__main__":
    main()
