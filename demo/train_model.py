import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ==============================
# 1. Load dataset
# ==============================

df = pd.read_csv("performance_prediction_output.csv")

# Clean column names
df.columns = [c.strip().replace(" ", "_") for c in df.columns]


# ==============================
# 2. Create performance score
# ==============================

df["performance_score"] = (
        (df["task_completion_rate"] * 30) +
        (df["on_time_delivery_rate"] * 25) +
        ((df["quality_score"] / 10) * 25) +
        (df["issue_resolution_rate"].clip(0, 1.2) / 1.2 * 10) -
        (df["rework_count"] * 1.0) -
        (df["blockers_count"] * 0.7) -
        (df["no_nopay_leave"] * 0.5)
)

df["performance_score"] = df["performance_score"].clip(0, 100).round(2)


# ==============================
# 3. Convert score to category
# ==============================

def classify_score(score):
    if score >= 75:
        return "High"
    elif score >= 55:
        return "Medium"
    else:
        return "Low"


df["performance_class"] = df["performance_score"].apply(classify_score)


# ==============================
# 4. Select model features
# ==============================

feature_cols = [
    "institution_id",
    "project_id",
    "job_role_id",
    "years_of_experience",
    "department",
    "tasks_assigned",
    "tasks_completed",
    "task_completion_rate",
    "tasks_on_time",
    "on_time_delivery_rate",
    "defect_count",
    "issue_resolution_rate",
    "quality_score",
    "rework_count",
    "blockers_count",
    "no_nopay_leave",
    "average_response_time",
    "meetings_attended",
    "learning_hours",
    "team_interaction_frequency"
]

X = df[feature_cols]
y = df["performance_score"]


# ==============================
# 5. Preprocessing
# ==============================

categorical_features = [
    "institution_id",
    "project_id",
    "job_role_id",
    "department"
]

numeric_features = [
    col for col in feature_cols
    if col not in categorical_features
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ]
)


# ==============================
# 6. Train model
# ==============================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=8
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model.fit(X_train, y_train)


# ==============================
# 7. Evaluate model
# ==============================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation")
print("----------------")
print("MAE:", round(mae, 4))
print("R2 Score:", round(r2, 4))


# ==============================
# 8. Predict for all employees
# ==============================

df["predicted_performance_score"] = model.predict(X).round(2)
df["predicted_performance_class"] = df["predicted_performance_score"].apply(classify_score)


# ==============================
# 9. Save model and output
# ==============================

joblib.dump(model, "performance_prediction_model.pkl")
df.to_csv("performance_prediction_output.csv", index=False)

print("\nSaved files:")
print("performance_prediction_model.pkl")
print("performance_prediction_output.csv")