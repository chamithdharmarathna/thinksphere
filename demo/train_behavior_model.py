import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline


# ==============================
# 1. Load Behavior Dataset
# ==============================

df = pd.read_csv("behavior_pattern_final_dataset.csv")
df.columns = [c.strip().replace(" ", "_") for c in df.columns]


# ==============================
# 2. Select Features
# ==============================

feature_cols = [
    "avg_punctuality",
    "avg_problem_solving",
    "avg_leadership",
    "avg_collaboration",
    "avg_communication",
    "avg_overall_behavior_rating",
    "manager_behavior_rating",
    "peer_behavior_rating",
    "subordinate_behavior_rating",
    "rating_variance",
    "sentiment_proxy_score",
    "communication_keyword_score",
    "leadership_keyword_score",
    "collaboration_keyword_score",
    "learning_hours",
    "meetings_attended",
    "average_response_time",
    "team_interaction_frequency"
]

# Convert selected features to numeric
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing values
df[feature_cols] = df[feature_cols].fillna(
    df[feature_cols].median(numeric_only=True)
)

X = df[feature_cols]


# ==============================
# 3. Find Best Cluster Count
# ==============================

scores = {}
models = {}

for k in [3, 4, 5, 6]:
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        ))
    ])

    labels = model.fit_predict(X)

    X_scaled = model.named_steps["scaler"].transform(X)
    score = silhouette_score(X_scaled, labels)

    scores[k] = score
    models[k] = model


best_k = max(scores, key=scores.get)
best_model = models[best_k]

print("Silhouette Scores:", scores)
print("Best Number of Clusters:", best_k)


# ==============================
# 4. Assign Clusters
# ==============================

df["behavior_cluster"] = best_model.predict(X)


# ==============================
# 5. Cluster Summary
# ==============================

cluster_summary = df.groupby("behavior_cluster")[feature_cols].mean().round(2)

print("\nCluster Summary:")
print(cluster_summary)


# ==============================
# 6. Assign Readable Cluster Labels
# ==============================

def assign_cluster_label(row):
    if row["avg_overall_behavior_rating"] >= 4.1 and row["avg_collaboration"] >= 4:
        return "Collaborative High Performer"

    elif row["avg_leadership"] >= 4 and row["avg_overall_behavior_rating"] >= 3.6:
        return "Emerging Leader"

    elif row["avg_communication"] <= 3 or row["average_response_time"] >= 45:
        return "Communication Improvement Needed"

    elif row["avg_collaboration"] <= 3:
        return "Collaboration Support Needed"

    elif row["avg_problem_solving"] >= 4:
        return "Technical Problem Solver"

    else:
        return "Balanced Contributor"


cluster_labels = {}

for cluster_id, row in cluster_summary.iterrows():
    cluster_labels[cluster_id] = assign_cluster_label(row)

df["behavior_cluster_label"] = df["behavior_cluster"].map(cluster_labels)

print("\nCluster Labels:")
print(cluster_labels)


# ==============================
# 7. PCA for Visualization
# ==============================

pca = PCA(n_components=2, random_state=42)

X_scaled = best_model.named_steps["scaler"].transform(X)
pca_values = pca.fit_transform(X_scaled)

df["pca_component_1"] = pca_values[:, 0].round(4)
df["pca_component_2"] = pca_values[:, 1].round(4)


# ==============================
# 8. Save Model and Outputs
# ==============================

joblib.dump(best_model, "behavior_clustering_model.pkl")
joblib.dump(pca, "behavior_pca_model.pkl")

df.to_csv("behavior_pattern_output.csv", index=False)
cluster_summary.to_csv("behavior_cluster_summary.csv", index=True)

print("\nSaved files:")
print("behavior_clustering_model.pkl")
print("behavior_pca_model.pkl")
print("behavior_pattern_output.csv")
print("behavior_cluster_summary.csv")