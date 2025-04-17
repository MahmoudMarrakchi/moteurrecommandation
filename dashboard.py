import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import locale

# Essayer de définir la locale en français (pour les dates et nombres)
try:
    locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'fr_FR')
    except:
        pass  # En cas d'échec, continuer avec la locale par défaut

# Configuration de la page
st.set_page_config(
    page_title="Analyse d'Engagement Utilisateur",
    page_icon="📊",
    layout="wide"
)

# Supprimer les avertissements spécifiques sur les opérations en virgule flottante
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Titre et description
st.title("Tableau de Bord d'Analyse d'Engagement Utilisateur")
st.markdown("""
Cette application analyse les données d'engagement des utilisateurs pour identifier des segments, 
prédire l'attrition et générer des recommandations personnalisées pour différents groupes d'utilisateurs.
""")

# Téléchargement de fichier
uploaded_file = st.sidebar.file_uploader("Télécharger CSV d'Engagement Utilisateur", type=["csv"])

# Charger les données
@st.cache_data
def load_data(file):
    if file is not None:
        data = pd.read_csv(file)
        return data
    else:
        # Utiliser des données d'exemple si aucun fichier n'est téléchargé
        st.sidebar.info("Utilisation des données d'exemple. Téléchargez votre propre CSV pour une analyse personnalisée.")
        # Essayer de charger le fichier par défaut si disponible
        try:
            return pd.read_csv("user_engagement.csv")
        except FileNotFoundError:
            st.error("Fichier de données d'exemple introuvable. Veuillez télécharger vos données.")
            return None

user_engagement = load_data(uploaded_file)

# Fonction pour attribuer des noms significatifs aux clusters
def nommer_clusters(df, kmeans_model):
    # Obtenir les centres des clusters
    cluster_centers = kmeans_model.cluster_centers_
    
    # Créer un dictionnaire pour stocker les caractéristiques dominantes de chaque cluster
    caracteristiques_clusters = {}
    
    # Déterminer la caractéristique dominante pour chaque cluster
    for i in range(len(cluster_centers)):
        # Trouver l'index de la valeur la plus élevée dans le centre du cluster
        dominant_feature_idx = np.argmax(cluster_centers[i])
        
        # Mapper à l'attribut correspondant
        features = ['frequency_score', 'interaction_score', 'loyalty_score', 'recency_score', 'engagement_score']
        dominant_feature = features[dominant_feature_idx]
        
        # Déterminer le nom du segment en fonction des caractéristiques
        if dominant_feature == 'recency_score' and cluster_centers[i][dominant_feature_idx] > 0.6:
            nom = "Utilisateurs Actifs Récemment"
        elif dominant_feature == 'frequency_score' and cluster_centers[i][dominant_feature_idx] > 0.6:
            nom = "Utilisateurs Fréquents"
        elif dominant_feature == 'loyalty_score' and cluster_centers[i][dominant_feature_idx] > 0.6:
            nom = "Utilisateurs Fidèles"
        elif dominant_feature == 'engagement_score' and cluster_centers[i][dominant_feature_idx] > 0.6:
            nom = "Utilisateurs Très Engagés"
        elif dominant_feature == 'interaction_score' and cluster_centers[i][dominant_feature_idx] > 0.6:
            nom = "Utilisateurs Interactifs"
        else:
            # Si aucune caractéristique n'est particulièrement élevée
            if np.mean(cluster_centers[i]) < 0.3:
                nom = "Utilisateurs Peu Engagés"
            else:
                nom = f"Segment {i+1}"
        
        caracteristiques_clusters[i] = nom
    
    # Créer un mapping des clusters numériques aux noms
    cluster_mapping = {i: nom for i, nom in caracteristiques_clusters.items()}
    
    # Appliquer le mapping pour créer une nouvelle colonne avec les noms des clusters
    df['nom_cluster'] = df['cluster'].map(cluster_mapping)
    
    return df, cluster_mapping

if user_engagement is not None:
    # Afficher les données brutes dans une section extensible
    with st.expander("Voir les Données Brutes"):
        st.dataframe(user_engagement)
    
    # Contrôles de la barre latérale
    st.sidebar.header("Paramètres d'Analyse")
    
    # Permettre à l'utilisateur d'ajuster les seuils de segmentation
    recency_threshold = st.sidebar.slider(
        "Seuil de Score de Récence", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7,
        help="Des valeurs plus élevées signifient que les utilisateurs doivent avoir visité plus récemment pour être considérés comme 'actifs'"
    )
    
    frequency_threshold = st.sidebar.slider(
        "Seuil de Score de Fréquence", 
        min_value=0.0, 
        max_value=1.0, 
        value=user_engagement['frequency_score'].median(),
        help="Des valeurs plus élevées signifient que les utilisateurs ont besoin d'interactions plus fréquentes pour être considérés comme 'actifs'"
    )
    
    churn_days = st.sidebar.slider(
        "Définition de l'Attrition (Jours d'Inactivité)", 
        min_value=1, 
        max_value=30, 
        value=8,
        help="Nombre de jours d'inactivité pour être considéré comme perdu"
    )
    
    cluster_count = st.sidebar.slider(
        "Nombre de Clusters d'Utilisateurs", 
        min_value=2, 
        max_value=10, 
        value=5
    )
    
    # Analyse principale
    st.header("Segmentation des Utilisateurs")
    col1, col2 = st.columns(2)
    
    with col1:
        # Créer des segments RFM
        user_engagement['segment'] = 'Inconnu'
        user_engagement.loc[(user_engagement['recency_score'] >= recency_threshold) & 
                          (user_engagement['frequency_score'] >= frequency_threshold), 'segment'] = 'Haute Valeur'
        user_engagement.loc[(user_engagement['recency_score'] < recency_threshold) & 
                          (user_engagement['frequency_score'] >= frequency_threshold), 'segment'] = 'À Risque'
        user_engagement.loc[(user_engagement['recency_score'] >= recency_threshold) & 
                          (user_engagement['frequency_score'] < frequency_threshold), 'segment'] = 'Nouveau/Potentiel'
        user_engagement.loc[(user_engagement['recency_score'] < recency_threshold) & 
                          (user_engagement['frequency_score'] < frequency_threshold), 'segment'] = 'Dormant'
        
        # Afficher les comptages par segment
        segment_counts = user_engagement['segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Nombre']
        st.write("Distribution des Segments d'Utilisateurs")
        st.dataframe(segment_counts)
        
        # Créer un graphique en camembert pour les segments
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(segment_counts['Nombre'], labels=segment_counts['Segment'], autopct='%1.1f%%',
               colors=sns.color_palette('viridis', n_colors=len(segment_counts)))
        ax.set_title('Distribution des Segments d\'Utilisateurs')
        st.pyplot(fig)
    
    with col2:
        # Graphique de segmentation Récence-Fréquence
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(x='recency_score', y='frequency_score', hue='segment', 
                        data=user_engagement, palette='viridis', s=100, alpha=0.7, ax=ax)
        
        # Ajouter les lignes de quadrant
        ax.axvline(x=recency_threshold, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=frequency_threshold, color='gray', linestyle='--', alpha=0.5)
        
        # Ajouter les étiquettes de segment
        ax.annotate('Haute Valeur', xy=(0.75, 0.75), xycoords='axes fraction', fontsize=14)
        ax.annotate('Nouveau/Potentiel', xy=(0.75, 0.25), xycoords='axes fraction', fontsize=14)
        ax.annotate('À Risque', xy=(0.25, 0.75), xycoords='axes fraction', fontsize=14)
        ax.annotate('Dormant', xy=(0.25, 0.25), xycoords='axes fraction', fontsize=14)
        
        ax.set_title('Segmentation des Utilisateurs par Récence et Fréquence', fontsize=16)
        ax.set_xlabel('Score de Récence (plus élevé = plus récent)', fontsize=14)
        ax.set_ylabel('Score de Fréquence (plus élevé = plus fréquent)', fontsize=14)
        st.pyplot(fig)
    
    # Analyse de Cluster
    st.header("Clustering Avancé des Utilisateurs")
    
    # Préparer les données pour le clustering
    clustering_features = ['frequency_score', 'interaction_score', 
                         'loyalty_score', 'recency_score', 'engagement_score']
    
    feature_names_fr = {
        'frequency_score': 'Score de Fréquence',
        'interaction_score': 'Score d\'Interaction',
        'loyalty_score': 'Score de Fidélité',
        'recency_score': 'Score de Récence',
        'engagement_score': 'Score d\'Engagement'
    }
    
    X = user_engagement[clustering_features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Appliquer le clustering K-means
    kmeans = KMeans(n_clusters=cluster_count, random_state=0)
    user_engagement['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Nommer les clusters
    user_engagement, cluster_mapping = nommer_clusters(user_engagement, kmeans)
    
    # Visualiser les clusters avec PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Tracer les clusters
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=user_engagement['cluster'], 
                         cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    ax.set_title('Visualisation des Segments Utilisateurs (ACP)', fontsize=15)
    ax.set_xlabel('Composante Principale 1', fontsize=12)
    ax.set_ylabel('Composante Principale 2', fontsize=12)
    
    # Ajouter des annotations pour les noms des clusters
    cluster_centers_pca = pca.transform(kmeans.cluster_centers_)
    for i, (x, y) in enumerate(cluster_centers_pca):
        ax.annotate(
            cluster_mapping[i], 
            (x, y),
            fontsize=12,
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    st.pyplot(fig)
    
    # Afficher les profils de cluster
    st.subheader("Profils des Clusters")
    
    # Créer un DataFrame avec les profils moyens des clusters
    cluster_profiles = user_engagement.groupby('nom_cluster')[clustering_features].mean().reset_index()
    
    # Renommer les colonnes pour l'affichage en français
    cluster_profiles_fr = cluster_profiles.copy()
    cluster_profiles_fr.columns = ['Nom du Cluster'] + [feature_names_fr[col] for col in clustering_features]
    
    st.dataframe(cluster_profiles_fr)
    
    # Graphique radar pour la comparaison des clusters
    st.subheader("Comparaison des Clusters")
    
    # Préparer les données pour le graphique radar
    cluster_profiles_norm = cluster_profiles.copy()
    for feature in clustering_features:
        min_val = cluster_profiles[feature].min()
        max_val = cluster_profiles[feature].max()
        cluster_profiles_norm[feature] = (cluster_profiles[feature] - min_val) / (max_val - min_val)
    
    # Sélectionner le cluster à visualiser
    selected_cluster_name = st.selectbox("Sélectionner un Cluster à Visualiser", 
                                        options=sorted(user_engagement['nom_cluster'].unique()))
    
    # Obtenir l'ID du cluster à partir du nom
    selected_cluster = None
    for cluster_id, name in cluster_mapping.items():
        if name == selected_cluster_name:
            selected_cluster = cluster_id
            break
    
    if selected_cluster is not None:
        # Créer un graphique radar
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Nombre de variables
        categories = clustering_features
        categories_fr = [feature_names_fr[cat] for cat in categories]
        N = len(categories)
        
        # Créer des angles pour chaque caractéristique
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Fermer la boucle
        
        # Obtenir les données pour le cluster sélectionné
        cluster_row = cluster_profiles_norm[cluster_profiles_norm['nom_cluster'] == selected_cluster_name]
        if not cluster_row.empty:
            values = cluster_row[categories].values.flatten().tolist()
            values += values[:1]  # Fermer la boucle
            
            # Dessiner le graphique
            ax.plot(angles, values, linewidth=2, linestyle='solid')
            ax.fill(angles, values, alpha=0.1)
            
            # Définir les étiquettes
            plt.xticks(angles[:-1], categories_fr, size=12)
            
            # Définir les graduations y
            ax.set_yticks([0.25, 0.5, 0.75])
            ax.set_yticklabels(['0.25', '0.5', '0.75'])
            
            ax.set_title(f"Profil pour le Cluster: {selected_cluster_name}", size=15)
            st.pyplot(fig)


    st.header("Analyse des Niveaux d'Engagement")

    # Créer les niveaux d'engagement
    try:
        user_engagement['niveau_engagement'] = pd.qcut(
            user_engagement['engagement_score'], 
            q=5, 
            labels=['Très Faible', 'Faible', 'Moyen', 'Élevé', 'Très Élevé'],
            duplicates='drop'  # Gestion des valeurs de limite dupliquées
        )
    except:
        # Approche alternative si la méthode ci-dessus échoue
        user_engagement['niveau_engagement'] = pd.cut(
            user_engagement['engagement_score'],
            bins=5,
            labels=['Très Faible', 'Faible', 'Moyen', 'Élevé', 'Très Élevé']
        )

    # Afficher la distribution des niveaux d'engagement
    st.subheader("Distribution des Niveaux d'Engagement")
    engagement_counts = user_engagement['niveau_engagement'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=engagement_counts.index, y=engagement_counts.values, ax=ax)
    ax.set_title("Distribution des Niveaux d'Engagement", fontsize=15)
    ax.set_xlabel("Niveau d'Engagement")
    ax.set_ylabel("Nombre d'Utilisateurs")
    st.pyplot(fig)

    # Créer la matrice de recommandation
    st.subheader("Matrice Segment vs Niveau d'Engagement")
    engagement_rec_matrix = pd.crosstab(user_engagement['niveau_engagement'], 
                                    user_engagement['segment'])

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(engagement_rec_matrix, annot=True, cmap='YlGnBu', fmt='d', ax=ax)
    ax.set_title('Matrice Niveau d\'Engagement vs Segment', fontsize=15)
    ax.set_ylabel('Niveau d\'Engagement')
    ax.set_xlabel('Segment Utilisateur')
    plt.tight_layout()
    st.pyplot(fig)

    
    
    # Analyse d'Attrition
    st.header("Prédiction d'Attrition")
    
    # Définir "attrition" comme les utilisateurs qui ne sont pas revenus depuis X jours
    user_engagement['attrition'] = user_engagement['days_since_last_activity'] > churn_days
    
    # Afficher le taux d'attrition
    churn_rate = user_engagement['attrition'].mean() * 100
    st.metric("Taux d'Attrition Global", f"{churn_rate:.1f}%")
    
    # Attrition par segment
    st.subheader("Attrition par Segment")
    churn_by_segment = user_engagement.groupby('segment')['attrition'].mean().reset_index()
    churn_by_segment['pourcentage_attrition'] = churn_by_segment['attrition'] * 100
    churn_by_segment.columns = ['Segment', 'Attrition', 'Pourcentage d\'Attrition']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Segment', y='Pourcentage d\'Attrition', data=churn_by_segment, ax=ax)
    ax.set_xlabel('Segment')
    ax.set_ylabel('Taux d\'Attrition (%)')
    ax.set_title('Taux d\'Attrition par Segment d\'Utilisateur')
    st.pyplot(fig)
    
    # Attrition par cluster nommé
    st.subheader("Attrition par Cluster")
    churn_by_cluster = user_engagement.groupby('nom_cluster')['attrition'].mean().reset_index()
    churn_by_cluster['pourcentage_attrition'] = churn_by_cluster['attrition'] * 100
    churn_by_cluster.columns = ['Cluster', 'Attrition', 'Pourcentage d\'Attrition']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Cluster', y='Pourcentage d\'Attrition', data=churn_by_cluster, ax=ax)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Taux d\'Attrition (%)')
    ax.set_title('Taux d\'Attrition par Cluster d\'Utilisateur')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Former le modèle de prédiction d'attrition
    if st.button("Former le Modèle de Prédiction d'Attrition"):
        with st.spinner("Formation du modèle en cours..."):
            X = user_engagement[clustering_features]
            y = user_engagement['attrition']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            
            # Former le modèle
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Évaluer
            y_pred = rf_model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Afficher les métriques
            st.subheader("Performance du Modèle")
            col1, col2, col3 = st.columns(3)
            col1.metric("Précision", f"{report['accuracy']:.2f}")
            col2.metric("Précision (Attrition)", f"{report[True]['precision']:.2f}")
            col3.metric("Rappel (Attrition)", f"{report[True]['recall']:.2f}")
            
            # Importance des caractéristiques
            feature_importance = pd.DataFrame({
                'Caractéristique': [feature_names_fr[feat] for feat in clustering_features],
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Caractéristique', data=feature_importance, ax=ax)
            ax.set_title('Caractéristiques Contribuant à la Prédiction d\'Attrition', fontsize=15)
            st.pyplot(fig)
    
    # Recommandations
    st.header("Recommandations pour les Utilisateurs")
    
    def generate_recommendations(user_data):
        recommendations = []
        
        # Pour les utilisateurs à haute valeur
        if user_data['segment'] == 'Haute Valeur':
            if user_data['days_since_last_activity'] > 7:
                recommendations.append('Réengager avec une newsletter personnalisée mettant en avant les dernières tendances en science des données')
            recommendations.append('Inviter à devenir un leader communautaire ou contributeur de contenu')
            recommendations.append('Offrir un accès anticipé aux fonctionnalités ou contenus premium')
        
        # Pour les utilisateurs à risque
        elif user_data['segment'] == 'À Risque':
            if user_data['interaction_score'] < 0.3:
                recommendations.append('Envoyer un sondage de contenu interactif pour stimuler l\'engagement')
            recommendations.append('Email personnalisé mettant en évidence le contenu populaire "manqué"')
            recommendations.append('Offre à durée limitée pour l\'essai de fonctionnalités premium')
        
        # Pour les nouveaux utilisateurs / potentiels
        elif user_data['segment'] == 'Nouveau/Potentiel':
            recommendations.append('Série de tutoriels sur les fonctionnalités de la plateforme')
            recommendations.append('Recommandations de contenu basées sur les modèles de navigation initiaux')
            recommendations.append('Récompense d\'engagement précoce (badge, reconnaissance)')
        
        # Pour les utilisateurs dormants
        else:  # Dormant
            recommendations.append('Campagne de réactivation avec incitations')
            if user_data['days_since_last_activity'] < 90:
                recommendations.append('Sondage pour comprendre les raisons du désengagement')
            else:
                recommendations.append('Offre spéciale "Vous nous manquez" avec un résumé de contenu personnalisé')
        
        return recommendations
    
    # Appliquer à tous les utilisateurs
    user_engagement['recommandations'] = user_engagement.apply(generate_recommendations, axis=1)
    
    # Montrer des recommandations pour un utilisateur exemple
    st.subheader("Exemples de Recommandations pour les Utilisateurs")
    
    # Permettre la sélection d'un segment d'utilisateur à afficher
    selected_segment = st.selectbox("Sélectionner un Segment", 
                                   options=user_engagement['segment'].unique())
    
    # Montrer un utilisateur exemple de ce segment
    sample_user = user_engagement[user_engagement['segment'] == selected_segment].sample(1).iloc[0]
    
    # Afficher le profil utilisateur
    st.write("Profil de l'Utilisateur:")
    user_profile = {
        'ID Utilisateur': sample_user.name,
        'Segment': sample_user['segment'],
        'Cluster': sample_user['nom_cluster'],
        'Score de Récence': f"{sample_user['recency_score']:.2f}",
        'Score de Fréquence': f"{sample_user['frequency_score']:.2f}",
        'Score d\'Engagement': f"{sample_user['engagement_score']:.2f}",
        'Jours Depuis Dernière Activité': sample_user['days_since_last_activity']
    }
    
    # Afficher en deux colonnes
    col1, col2 = st.columns(2)
    for i, (key, value) in enumerate(user_profile.items()):
        if i % 2 == 0:
            col1.metric(key, value)
        else:
            col2.metric(key, value)
    
    # Afficher les recommandations
    st.write("Actions Recommandées:")
    for rec in sample_user['recommandations']:
        st.success(rec)
    
    # Fonctionnalité d'exportation
    st.header("Exporter les Résultats")
    
    export_columns = st.multiselect(
        "Sélectionner les colonnes à exporter",
        options=list(user_engagement.columns),
        default=['segment', 'cluster', 'nom_cluster', 'attrition', 'recommandations']
    )
    
    if st.button("Générer l'Exportation"):
        export_df = user_engagement[['user_id'] + export_columns] if 'user_id' in user_engagement.columns else user_engagement[export_columns]
        
        # Pour la liste des recommandations, convertir en chaîne
        if 'recommandations' in export_df.columns:
            export_df['recommandations'] = export_df['recommandations'].apply(lambda x: '; '.join(x))
        
        # Renommer les colonnes en français pour l'export
        rename_dict = {
            'cluster': 'cluster_id',
            'nom_cluster': 'cluster',
            'attrition': 'attrition',
            'recommandations': 'recommandations'
        }
        export_df = export_df.rename(columns={k: v for k, v in rename_dict.items() if k in export_df.columns})
        
        # Créer un lien de téléchargement
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Télécharger CSV",
            data=csv,
            file_name="analyse_engagement_utilisateurs.csv",
            mime="text/csv"
        )
else:
    st.error("Aucune donnée disponible. Veuillez télécharger un fichier CSV d'engagement utilisateur.")


# Ajouter cette section après la partie "Exporter les Résultats"

# Nouvelle section pour prédire le segment utilisateur
st.header("Prédire le Segment Utilisateur")
st.markdown("""
Cette section vous permet d'entrer les métriques d'engagement d'un utilisateur et de prédire
son segment, son cluster, et d'obtenir des recommandations personnalisées.
""")

# Créer un formulaire pour les entrées
with st.form(key='prediction_form'):
    col1, col2 = st.columns(2)
    
    with col1:
        recency_score = st.slider(
            "Score de Récence",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Score indiquant la récence des visites (1.0 = très récent)"
        )
        
        frequency_score = st.slider(
            "Score de Fréquence",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Score indiquant la fréquence des interactions (1.0 = très fréquent)"
        )
        
        loyalty_score = st.slider(
            "Score de Fidélité",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Score indiquant la fidélité de l'utilisateur (1.0 = très fidèle)"
        )
    
    with col2:
        interaction_score = st.slider(
            "Score d'Interaction",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Score indiquant le niveau d'interaction de l'utilisateur (1.0 = très interactif)"
        )
        
        engagement_score = st.slider(
            "Score d'Engagement",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Score global d'engagement (1.0 = très engagé)"
        )
        
        days_since_last_activity = st.number_input(
            "Jours depuis la dernière activité",
            min_value=0,
            max_value=365,
            value=5,
            help="Nombre de jours écoulés depuis la dernière activité de l'utilisateur"
        )
    
    submit_button = st.form_submit_button(label='Prédire le Segment')

# Traitement lorsque le formulaire est soumis
if submit_button and user_engagement is not None:
    # Créer un DataFrame pour l'utilisateur à prédire
    user_data = pd.DataFrame({
        'recency_score': [recency_score],
        'frequency_score': [frequency_score],
        'loyalty_score': [loyalty_score],
        'interaction_score': [interaction_score],
        'engagement_score': [engagement_score],
        'days_since_last_activity': [days_since_last_activity]
    })
    
    # Déterminer le segment RFM
    if recency_score >= recency_threshold and frequency_score >= frequency_threshold:
        predicted_segment = 'Haute Valeur'
    elif recency_score < recency_threshold and frequency_score >= frequency_threshold:
        predicted_segment = 'À Risque'
    elif recency_score >= recency_threshold and frequency_score < frequency_threshold:
        predicted_segment = 'Nouveau/Potentiel'
    else:
        predicted_segment = 'Dormant'
    
    # Prédire le cluster
    X_user = user_data[clustering_features].values
    X_user_scaled = scaler.transform(X_user)
    predicted_cluster = kmeans.predict(X_user_scaled)[0]
    cluster_name = cluster_mapping[predicted_cluster]
    
    # Déterminer si l'utilisateur est à risque d'attrition
    attrition_risk = days_since_last_activity > churn_days
    
    # Générer les recommandations
    user_data['segment'] = predicted_segment
    recommendations = generate_recommendations(user_data.iloc[0])
    
    # Afficher les résultats
    st.subheader("Résultats de la Prédiction")
    
    # Afficher les métriques principales
    col1, col2, col3 = st.columns(3)
    col1.metric("Segment Prédit", predicted_segment)
    col2.metric("Cluster Prédit", cluster_name)
    col3.metric("Risque d'Attrition", "Élevé" if attrition_risk else "Faible")
    
    # Afficher les recommandations
    st.subheader("Recommandations Personnalisées")
    for rec in recommendations:
        st.success(rec)
    
    # Visualiser la position de l'utilisateur sur le graphique de segmentation
    st.subheader("Visualisation de la Position de l'Utilisateur")
    
    # Recréer le graphique de segmentation avec le nouvel utilisateur
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Tracer les utilisateurs existants (en arrière-plan)
    sns.scatterplot(x='recency_score', y='frequency_score', hue='segment', 
                    data=user_engagement, palette='viridis', s=50, alpha=0.3, ax=ax)
    
    # Tracer le nouvel utilisateur (en évidence)
    ax.scatter(recency_score, frequency_score, color='red', s=200, marker='*', label='Utilisateur Prédit')
    
    # Ajouter les lignes de quadrant
    ax.axvline(x=recency_threshold, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=frequency_threshold, color='gray', linestyle='--', alpha=0.5)
    
    # Ajouter les étiquettes de segment
    ax.annotate('Haute Valeur', xy=(0.75, 0.75), xycoords='axes fraction', fontsize=14)
    ax.annotate('Nouveau/Potentiel', xy=(0.75, 0.25), xycoords='axes fraction', fontsize=14)
    ax.annotate('À Risque', xy=(0.25, 0.75), xycoords='axes fraction', fontsize=14)
    ax.annotate('Dormant', xy=(0.25, 0.25), xycoords='axes fraction', fontsize=14)
    
    # Configuration du graphique
    ax.set_title('Position de l\'Utilisateur dans la Segmentation', fontsize=16)
    ax.set_xlabel('Score de Récence (plus élevé = plus récent)', fontsize=14)
    ax.set_ylabel('Score de Fréquence (plus élevé = plus fréquent)', fontsize=14)
    ax.legend()
    
    st.pyplot(fig)
    
    # Visualiser la position dans l'espace des clusters (ACP)
    st.subheader("Position dans l'Espace des Clusters")
    
    # Transformer les données utilisateur avec PCA
    user_pca = pca.transform(X_user_scaled)
    
    # Créer un graphique avec les clusters existants et le nouvel utilisateur
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Tracer les clusters existants
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=user_engagement['cluster'], 
                         cmap='viridis', s=50, alpha=0.3)
    
    # Tracer le nouvel utilisateur
    ax.scatter(user_pca[0, 0], user_pca[0, 1], color='red', s=200, marker='*', label='Utilisateur Prédit')
    
    # Ajouter les centres des clusters et leurs noms
    for i, (x, y) in enumerate(cluster_centers_pca):
        ax.annotate(
            cluster_mapping[i], 
            (x, y),
            fontsize=12,
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    # Configuration du graphique
    ax.set_title('Position de l\'Utilisateur dans l\'Espace des Clusters', fontsize=15)
    ax.set_xlabel('Composante Principale 1', fontsize=12)
    ax.set_ylabel('Composante Principale 2', fontsize=12)
    ax.legend()
    
    st.pyplot(fig)

# Pied de page
st.sidebar.markdown("---")
st.sidebar.info("""
**À propos de cette application**  
Ce tableau de bord Streamlit analyse les données d'engagement des utilisateurs pour identifier 
les segments d'utilisateurs, prédire l'attrition et générer des recommandations personnalisées.
""")