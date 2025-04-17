import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import warnings

# Suppress specific warnings about floating point operations
warnings.filterwarnings('ignore', category=RuntimeWarning)
# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

user_engagement = pd.read_csv("user_engagement.csv")

# Create a visualization for user segmentation based on recency and frequency
plt.figure(figsize=(12, 10))

# Calculate quantiles for recency and frequency
# recency_median = user_engagement['recency_score'].median()
frequency_median = user_engagement['frequency_score'].median()
recency_median = 0.7
# Create RFM segments
user_engagement['segment'] = 'Unknown'
user_engagement.loc[(user_engagement['recency_score'] >= recency_median) & 
                    (user_engagement['frequency_score'] >= frequency_median), 'segment'] = 'High Value'
user_engagement.loc[(user_engagement['recency_score'] < recency_median) & 
                    (user_engagement['frequency_score'] >= frequency_median), 'segment'] = 'At Risk'
user_engagement.loc[(user_engagement['recency_score'] >= recency_median) & 
                    (user_engagement['frequency_score'] < frequency_median), 'segment'] = 'New/Potential'
user_engagement.loc[(user_engagement['recency_score'] < recency_median) & 
                    (user_engagement['frequency_score'] < frequency_median), 'segment'] = 'Dormant'

# Plot the segments
sns.scatterplot(x='recency_score', y='frequency_score', hue='segment', 
                data=user_engagement, palette='viridis', s=100, alpha=0.7)

# Add quadrant lines
plt.axvline(x=recency_median, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=frequency_median, color='gray', linestyle='--', alpha=0.5)

# Add segment labels
plt.annotate('High Value', xy=(0.75, 0.75), xycoords='axes fraction', fontsize=14)
plt.annotate('New/Potential', xy=(0.75, 0.25), xycoords='axes fraction', fontsize=14)
plt.annotate('At Risk', xy=(0.25, 0.75), xycoords='axes fraction', fontsize=14)
plt.annotate('Dormant', xy=(0.25, 0.25), xycoords='axes fraction', fontsize=14)

plt.title('User Segmentation by Recency and Frequency', fontsize=16)
plt.xlabel('Recency Score (higher = more recent)', fontsize=14)
plt.ylabel('Frequency Score (higher = more frequent)', fontsize=14)
plt.show()

# Prepare data for clustering
clustering_features = ['frequency_score', 'interaction_score', 
                       'loyalty_score', 'recency_score', 'engagement_score']
                       
X = user_engagement[clustering_features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0)
user_engagement['cluster'] = kmeans.fit_predict(X_scaled)

# Visualize clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=user_engagement['cluster'], cmap='viridis', s=50, alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title('User Segments Visualization (PCA)', fontsize=15)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.show()


# Define "churn" as users who haven't returned in 30 days
user_engagement['churned'] = user_engagement['days_since_last_activity'] > 8

# Prepare features and target
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

X = user_engagement[clustering_features]
y = user_engagement['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': clustering_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Features Contributing to Churn Prediction', fontsize=15)
plt.tight_layout()
plt.show()


def generate_recommendations(user_data):
    recommendations = []
    
    # For high-value users
    if user_data['segment'] == 'High Value':
        if user_data['days_since_last_activity'] > 7:
            recommendations.append('Re-engage with personalized newsletter highlighting latest data science trends')
        recommendations.append('Invite to become a community leader or content contributor')
        recommendations.append('Offer early access to premium features or content')
    
    # For at-risk users
    elif user_data['segment'] == 'At Risk':
        if user_data['interaction_score'] < 0.3:
            recommendations.append('Send interactive content poll to boost engagement')
        recommendations.append('Personalized email highlighting "missed" popular content')
        recommendations.append('Limited-time offer for premium feature trial')
    
    # For new/potential users
    elif user_data['segment'] == 'New/Potential':
        recommendations.append('Tutorial series on platform features')
        recommendations.append('Content recommendations based on initial browsing patterns')
        recommendations.append('Early engagement reward (badge, recognition)')
    
    # For dormant users
    else:  # Dormant
        recommendations.append('Re-activation campaign with incentives')
        if user_data['days_since_last_activity'] < 90:
            recommendations.append('Survey to understand disengagement reasons')
        else:
            recommendations.append('Special "We miss you" offer with personalized content digest')
    
    return recommendations

# Apply to all users
user_engagement['recommendations'] = user_engagement.apply(generate_recommendations, axis=1)