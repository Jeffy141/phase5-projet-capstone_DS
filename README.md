# Projet Capstone – Phase 5
## Système de Maintenance Prédictive pour Équipements IoT

### Description du projet
Ce projet de phase 5 vise à concevoir un modèle de machine learning pour prédire les pannes d’équipements électroniques IoT à partir de données capteurs. 
Il intègre un pipeline complet de traitement et d’analyse de données industrielles, avec déploiement d’une interface interactive et d’une API.

### Objectifs
- Construire un pipeline ML complet (prétraitement, modélisation, interprétation)
- Détecter les défaillances à partir des mesures physiques
- Évaluer les performances avec précision, rappel, F1 et ROC AUC
- Déployer un système utilisable via Streamlit et Flask

### Données
Le dataset comporte environ 50 000 lignes et 17 colonnes :
- Variables physiques : Température, Vibration, Pression, Tension, Courant
- Variables dérivées : FFT, Score_Anomalie, Normalisations
- Cible : Fault_Status (0 = normal, 1 = défaut)

### Méthodologie
1. Exploration et nettoyage des données
2. Feature engineering et extraction temporelle
3. Modélisation : RandomForest, LogisticRegression
4. Validation croisée et tuning d’hyperparamètres
5. Interprétation SHAP et déploiement (Streamlit + Flask)

### Résultats
| Modèle | Accuracy | Precision | Recall | F1 | ROC AUC |
|--------|-----------|------------|--------|----|----------|
| Random Forest | 0.93 | 0.89 | 0.91 | 0.90 | 0.95 |
| Logistic Regression | 0.88 | 0.84 | 0.86 | 0.85 | 0.90 |

### Déploiement
- Dashboard : `streamlit run app.py`
- API REST : `python api.py`
- Dépendances : `pip install -r requirements.txt`

### Structure du dépôt
```
├── iot_equipment_monitoring_notebook_v1.ipynb
├── models/
│   ├── final_pipeline.joblib
│   ├── features_list.json
│   └── test_metrics.json
├── app.py
├── api.py
├── requirements.txt
├── presentation.pdf
└── README.md
```

### Conclusion
Ce projet illustre la maîtrise d’un pipeline complet de Data Science appliqué à la maintenance prédictive. 
Les résultats démontrent une forte fiabilité et un potentiel d’application industrielle élevé.
