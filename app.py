import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import joblib
from datetime import datetime
from pycaret.classification import*
import warnings
import matplotlib.pyplot as plt

#from .registry import BackendFilter, backend_registry

warnings.filterwarnings('ignore')

# Configuration avancée de la page
st.set_page_config(
    page_title="ML Fraud Detection System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un look professionnel
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .risk-low {
        background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# =================================================================
# CHARGEMENT DU MODÈLE ENTRAÎNÉ - VOTRE VERSION INTÉGRÉE
# =================================================================
@st.cache_resource
def load_trained_model():
    """
    Chargement de votre modèle entraîné
    """
    try:
        # Chargement de votre modèle réel
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Si vous avez un scaler, chargez-le aussi
        # with open('scaler.pkl', 'rb') as f:
        #     scaler = pickle.load(f)
        
        return {
            'model': model,
            'scaler': None,  # Ajoutez votre scaler ici si nécessaire
            'status': 'real'
        }
    except FileNotFoundError:
        st.warning("⚠️ Modèle 'model.pkl' non trouvé. Utilisation d'un modèle de démonstration.")
        # Fallback sur modèle de démo
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        X_dummy = np.random.rand(1000, 8)
        y_dummy = np.random.randint(0, 2, 1000)
        model.fit(X_dummy, y_dummy)
        
        return {
            'model': model,
            'scaler': None,
            'status': 'demo'
        }

# Chargement des artefacts ML
ml_artifacts = load_trained_model()
modele_rf = ml_artifacts['model']
scaler = ml_artifacts['scaler']
model_status = ml_artifacts['status']

# Noms des features
feature_names = ['age', 'salaire', 'score_credit', 'montant_transaction', 
                'anciennete_compte', 'type_carte', 'genre', 'region_encode']
# =================================================================

def preprocess_data(data):
    """
    Préprocessing des données d'entrée
    Version améliorée avec gestion du scaler
    """
    df = pd.DataFrame([data], columns=feature_names)
    
    # Appliquer le scaler si disponible
    if scaler is not None:
        return scaler.transform(df.values)
    else:
        return df.values

def predict_fraud(data):
    """
    Prédiction de fraude améliorée
    """
    processed_data = preprocess_data(data)
    prediction = modele_rf.predict(processed_data)[0]
    probability = modele_rf.predict_proba(processed_data)[0]
    
    return prediction, probability

def calculate_advanced_risk_score(prediction_proba, feature_values):
    """
    Score de risque avancé avec règles métier
    """
    base_score = prediction_proba[1] * 100
    
    # Facteurs d'ajustement métier
    age_factor = 1.1 if feature_values[0] < 25 or feature_values[0] > 65 else 1.0
    amount_factor = 1.15 if feature_values[3] > 1000 else 1.0
    credit_factor = 1.2 if feature_values[2] < 500 else 1.0
    
    final_score = base_score * age_factor * amount_factor * credit_factor
    return min(final_score, 100)

def create_feature_radar_chart(feature_values):
    """
    Graphique radar des caractéristiques client
    """
    # Normalisation pour visualisation
    ranges = {
        'age': (18, 80), 'salaire': (20000, 100000), 'score_credit': (300, 850),
        'montant_transaction': (0, 2000), 'anciennete_compte': (0, 120),
        'type_carte': (0, 2), 'genre': (0, 1), 'region_encode': (0, 9)
    }
    
    normalized_values = []
    for i, (feature, value) in enumerate(zip(feature_names, feature_values)):
        min_val, max_val = ranges[feature]
        normalized = (value - min_val) / (max_val - min_val) * 100
        normalized_values.append(max(0, min(100, normalized)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=feature_names,
        fill='toself',
        name='Profil Client',
        line_color='rgba(1, 87, 155, 0.8)',
        fillcolor='rgba(1, 87, 155, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Profil Radar du Client"
    )
    return fig

def generate_feature_explanation(feature_values):
    """
    Explication simplifiée de l'impact des features
    """
    explanations = []
    
    # Règles d'explication basées sur les valeurs
    if feature_values[0] < 25:
        explanations.append("👤 Âge jeune : risque légèrement élevé")
    elif feature_values[0] > 65:
        explanations.append("👴 Âge avancé : profil à surveiller")
    
    if feature_values[1] > 80000:
        explanations.append("💰 Salaire élevé : profil favorable")
    elif feature_values[1] < 30000:
        explanations.append("💸 Salaire faible : facteur de risque")
    
    if feature_values[2] < 500:
        explanations.append("📉 Score crédit faible : risque élevé")
    elif feature_values[2] > 750:
        explanations.append("📈 Excellent score crédit : très favorable")
    
    if feature_values[3] > 1000:
        explanations.append("💳 Transaction importante : surveillance requise")
    
    if feature_values[4] < 6:
        explanations.append("🆕 Compte récent : facteur de risque")
    
    return explanations

# Interface utilisateur principale
st.markdown('<h1 class="main-header">🤖 Advanced ML Fraud Detection System</h1>', unsafe_allow_html=True)

# Sidebar avec informations sur le modèle
with st.sidebar:
    st.header("🔬 Model Information")
    
    # Statut du modèle
    if model_status == 'real':
        st.success("✅ Modèle réel chargé")
    else:
        st.warning("⚠️ Modèle de démonstration")
    
    st.info(f"""
    **Algorithm:** Random Forest
    **Features:** {len(feature_names)}
    **Scaler:** {'✅ Loaded' if scaler else '❌ Not used'}
    """)
    
    # Informations sur les features
    st.header("📊 Features Info")
    for i, feature in enumerate(feature_names):
        st.write(f"**{i+1}.** {feature}")

# Interface principale avec onglets
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📈 Batch Analysis", "🎯 Model Insights"])

with tab1:
    st.header("🔍 Transaction Risk Assessment")
    
    # Interface de saisie améliorée
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Profil Client")
        age = st.slider("Âge", 18, 100, 35)
        salaire = st.number_input("Salaire (€)", 0, 200000, 45000, step=5000)
        score_credit = st.slider("Score de Crédit", 300, 850, 650)
    
    with col2:
        st.subheader("💳 Transaction")
        montant_transaction = st.number_input("Montant (€)", 0.0, 5000.0, 150.0, step=50.0)
        anciennete_compte = st.slider("Ancienneté (mois)", 0, 120, 24)
        type_carte = st.selectbox("Type de Carte", options=[0, 1, 2], 
                                 format_func=lambda x: ["Débit", "Crédit", "Prépayée"][x])
    
    with col3:
        st.subheader("🏷️ Informations")
        genre = st.selectbox("Genre", options=[0, 1], 
                           format_func=lambda x: ["Femme", "Homme"][x])
        region_encode = st.selectbox("Région", options=list(range(10)),
                                   format_func=lambda x: f"Région {x}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 Analyser Transaction", type="primary", 
                               use_container_width=True)
    
    if analyze_btn:
        # Préparer les données
        feature_values = [age, salaire, score_credit, montant_transaction, 
                         anciennete_compte, type_carte, genre, region_encode]
        
        # Prédiction
        prediction, probability = predict_fraud(feature_values)
        risk_score = calculate_advanced_risk_score(probability, feature_values)
        
        # Affichage des résultats principaux
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if prediction == 1:
                st.markdown('<div class="risk-high">🚨 FRAUDE DÉTECTÉE</div>', 
                           unsafe_allow_html=True)
            else:
                if risk_score > 30:
                    st.markdown('<div class="risk-medium">⚠️ RISQUE MOYEN</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown('<div class="risk-low">✅ FAIBLE RISQUE</div>', 
                               unsafe_allow_html=True)
        
        with col2:
            st.metric("Score de Risque", f"{risk_score:.1f}/100")
        
        with col3:
            st.metric("Prob. Fraude", f"{probability[1]:.1%}")
        
        with col4:
            confidence = max(probability)
            st.metric("Confiance", f"{confidence:.1%}")
        
        # Visualisations avancées
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique de probabilités
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Bar(
                x=['Légitime', 'Frauduleuse'],
                y=[probability[0], probability[1]],
                marker_color=['#4CAF50', '#F44336'],
                text=[f'{probability[0]:.1%}', f'{probability[1]:.1%}'],
                textposition='auto'
            ))
            fig_prob.update_layout(
                title="Probabilités de Prédiction",
                yaxis_title="Probabilité",
                showlegend=False
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        
        with col2:
            # Radar chart du profil
            radar_fig = create_feature_radar_chart(feature_values)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        # Explications des features
        st.subheader("🔍 Analyse des Facteurs")
        explanations = generate_feature_explanation(feature_values)
        
        if explanations:
            for explanation in explanations:
                st.write(f"• {explanation}")
        else:
            st.write("• Profil standard sans facteurs de risque particuliers")
        
        # Recommandations
        st.subheader("💡 Recommandations")
        
        if prediction == 1:
            st.error("""
            **🚨 Actions Immédiates :**
            • Bloquer temporairement la transaction
            • Contacter immédiatement le client
            • Examiner l'historique des 30 derniers jours
            • Activer les alertes de sécurité renforcées
            """)
        else:
            if probability[1] > 0.3:
                st.warning("""
                **⚠️ Surveillance Recommandée :**
                • Surveiller les prochaines transactions
                • Vérifier l'activité dans les 24h
                • Alertes automatiques activées
                """)
            else:
                st.success("""
                **✅ Transaction Approuvée :**
                • Aucune action particulière requise
                • Profil client considéré comme sûr
                • Traitement normal autorisé
                """)

with tab2:
    st.header("📈 Analyse par Lot")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Template CSV")
        template_df = pd.DataFrame({
            'age': [35, 42, 28, 55],
            'salaire': [45000, 62000, 38000, 75000],
            'score_credit': [650, 720, 580, 680],
            'montant_transaction': [150.0, 250.0, 75.0, 500.0],
            'anciennete_compte': [24, 36, 12, 48],
            'type_carte': [1, 0, 2, 1],
            'genre': [1, 0, 1, 0],
            'region_encode': [3, 5, 1, 7]
        })
        st.dataframe(template_df, use_container_width=True)
    
    with col2:
        st.subheader("📤 Upload & Analyse")
        uploaded_file = st.file_uploader("Fichier CSV", type="csv")
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"✅ {len(batch_df)} transactions chargées")
                
                if st.button("🚀 Analyser le Lot", type="primary"):
                    # Vérification des colonnes
                    missing_cols = set(feature_names) - set(batch_df.columns)
                    if missing_cols:
                        st.error(f"Colonnes manquantes: {missing_cols}")
                    else:
                        # Traitement batch
                        results = []
                        for _, row in batch_df.iterrows():
                            data = row[feature_names].values
                            pred, prob = predict_fraud(data)
                            risk = calculate_advanced_risk_score(prob, data)
                            results.append({
                                'prediction': pred,
                                'fraud_probability': prob[1],
                                'risk_score': risk
                            })
                        
                        # Créer DataFrame des résultats
                        results_df = batch_df.copy()
                        for i, result in enumerate(results):
                            results_df.loc[i, 'fraud_prediction'] = result['prediction']
                            results_df.loc[i, 'fraud_probability'] = result['fraud_probability']
                            results_df.loc[i, 'risk_score'] = result['risk_score']
                        
                        # Métriques du lot
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total", len(results_df))
                        
                        with col2:
                            fraud_count = sum(r['prediction'] for r in results)
                            st.metric("Fraudes", fraud_count, 
                                     delta=f"{fraud_count/len(results_df)*100:.1f}%")
                        
                        with col3:
                            high_risk = sum(1 for r in results if r['risk_score'] > 70)
                            st.metric("Haut Risque", high_risk)
                        
                        with col4:
                            avg_risk = np.mean([r['risk_score'] for r in results])
                            st.metric("Risque Moyen", f"{avg_risk:.1f}/100")
                        
                        # Affichage des résultats
                        st.subheader("📊 Résultats Détaillés")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Visualisation
                        risk_counts = pd.cut([r['risk_score'] for r in results], 
                                           bins=[0, 30, 70, 100], 
                                           labels=['Faible', 'Moyen', 'Élevé']).value_counts()
                        
                        fig_risk = px.pie(values=risk_counts.values, names=risk_counts.index,
                                         title="Distribution des Niveaux de Risque")
                        st.plotly_chart(fig_risk, use_container_width=True)
            
            except Exception as e:
                st.error(f"Erreur lors du traitement: {str(e)}")

with tab3:
    st.header("🎯 Insights du Modèle")
    
    if hasattr(modele_rf, 'feature_importances_'):
        # Importance des features
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': modele_rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_imp = px.bar(importance_df, x='Importance', y='Feature', 
                            orientation='h', title="Importance des Variables")
            st.plotly_chart(fig_imp, use_container_width=True)
        
        with col2:
            # Statistiques du modèle
            st.subheader("📊 Statistiques Modèle")
            st.write(f"**Nombre d'arbres:** {modele_rf.n_estimators}")
            st.write(f"**Profondeur max:** {getattr(modele_rf, 'max_depth', 'Non limitée')}")
            st.write(f"**Features utilisées:** {len(feature_names)}")
            
            # Top 3 features les plus importantes
            st.subheader("🏆 Top 3 Features")
            for i, (_, row) in enumerate(importance_df.head(3).iterrows()):
                st.write(f"**{i+1}.** {row['Feature']}: {row['Importance']:.3f}")
    
    else:
        st.warning("Informations d'importance non disponibles pour ce modèle")
    
    # Informations générales
    st.subheader("ℹ️ Informations Générales")
    st.write("""
    **Modèle:** Random Forest Classifier
    **Objectif:** Détection de fraudes bancaires
    **Type:** Classification binaire (Fraude/Légitime)
    **Méthode:** Ensemble d'arbres de décision
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>🤖 Advanced ML Fraud Detection System | 
    Powered by Random Forest | 
    Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
