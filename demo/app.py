import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# Page setup
# ==============================

st.set_page_config(
    page_title="AI-Based Employee Appraisal Demo",
    layout="wide"
)

st.title("AI-Based Employee Performance Appraisal System")
st.write(
    "Demo for Performance Prediction, Behavior Pattern Analysis, "
    "Bias Detection, and Decision Arbitration"
)


# ==============================
# Load datasets
# ==============================

performance_df = pd.read_csv("performance_prediction_output.csv")
behavior_df = pd.read_csv("behavior_pattern_final_dataset.csv")
bias_df = pd.read_csv("bias_detection_output.csv")
decision_df = pd.read_csv("decision_arbitration_dataset.csv")

# Clean column names
for df in [performance_df, behavior_df, bias_df, decision_df]:
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]


# ==============================
# Sidebar employee selection
# ==============================

st.sidebar.header("Employee Selection")

employee_id = st.sidebar.selectbox(
    "Select Employee ID",
    performance_df["employee_id"].unique()
)

performance_employee = performance_df[
    performance_df["employee_id"] == employee_id
    ].iloc[0]

behavior_employee = behavior_df[
    behavior_df["employee_id"] == employee_id
    ].iloc[0]

bias_employee = bias_df[
    bias_df["employee_id"] == employee_id
    ].iloc[0]

decision_employee = decision_df[
    decision_df["employee_id"] == employee_id
    ].iloc[0]


# ==============================
# Employee profile
# ==============================

st.header("Employee Profile")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Employee ID", employee_id)
col2.metric("Job Role", performance_employee["job_role_id"])
col3.metric("Department", performance_employee["department"])
col4.metric("Experience", f"{performance_employee['years_of_experience']} years")

col5, col6, col7, col8 = st.columns(4)

col5.metric("Gender", performance_employee["gender"])

if "age_group" in bias_df.columns:
    col6.metric("Age Group", bias_employee["age_group"])
else:
    col6.metric("Age Group", "N/A")

col7.metric("Institution", performance_employee["institution_id"])
col8.metric("Project", performance_employee["project_id"])


# ==============================
# Module 1: Performance Prediction
# ==============================

st.header("Module 1: Performance Prediction")

current_score = performance_employee["performance_score"]
predicted_score = performance_employee["predicted_performance_score"]
predicted_class = performance_employee["predicted_performance_class"]

col1, col2, col3 = st.columns(3)

col1.metric("Current Performance Score", round(current_score, 2))
col2.metric("Predicted Performance Score", round(predicted_score, 2))
col3.metric("Predicted Class", predicted_class)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(
    ["Current Score", "Predicted Score"],
    [current_score, predicted_score]
)
ax.set_ylim(0, 100)
ax.set_ylabel("Score")
ax.set_title("Current vs Predicted Performance Score")
st.pyplot(fig)

st.subheader("Key Performance Inputs")

performance_input_cols = [
    "task_completion_rate",
    "on_time_delivery_rate",
    "quality_score",
    "issue_resolution_rate",
    "rework_count",
    "blockers_count",
    "no_nopay_leave"
]

available_performance_cols = [
    col for col in performance_input_cols
    if col in performance_df.columns
]

st.dataframe(
    performance_employee[available_performance_cols].to_frame(name="Value")
)


# ==============================
# Module 2: Behavior Pattern Analysis
# ==============================

st.header("Module 2: Behavior Pattern Analysis")

col1, col2, col3 = st.columns(3)

col1.metric("Overall Behavior Pattern", behavior_employee["overall_behavior_pattern"])
col2.metric("Communication Pattern", behavior_employee["communication_pattern"])
col3.metric("Collaboration Pattern", behavior_employee["collaboration_pattern"])

col4, col5, col6 = st.columns(3)

col4.metric("Leadership Pattern", behavior_employee["leadership_pattern"])
col5.metric("Productivity Pattern", behavior_employee["productivity_pattern"])
col6.metric("Learning Pattern", behavior_employee["learning_pattern"])

st.subheader("Behavior Rating Summary")

behavior_scores = {
    "Punctuality": behavior_employee["avg_punctuality"],
    "Problem Solving": behavior_employee["avg_problem_solving"],
    "Leadership": behavior_employee["avg_leadership"],
    "Collaboration": behavior_employee["avg_collaboration"],
    "Communication": behavior_employee["avg_communication"],
    "Overall Behavior": behavior_employee["avg_overall_behavior_rating"]
}

behavior_score_df = pd.DataFrame(
    list(behavior_scores.items()),
    columns=["Behavior Attribute", "Average Rating"]
)

st.dataframe(behavior_score_df)

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.bar(
    behavior_score_df["Behavior Attribute"],
    behavior_score_df["Average Rating"]
)
ax2.set_ylim(0, 5)
ax2.set_ylabel("Average Rating")
ax2.set_title("Behavior Attribute Ratings")
plt.xticks(rotation=30, ha="right")
st.pyplot(fig2)

st.subheader("Manager, Peer, and Subordinate Behavior Ratings")

source_rating_df = pd.DataFrame({
    "Evaluator Source": ["Manager", "Peer", "Subordinate"],
    "Rating": [
        behavior_employee["manager_behavior_rating"],
        behavior_employee["peer_behavior_rating"],
        behavior_employee["subordinate_behavior_rating"]
    ]
})

st.dataframe(source_rating_df)

fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.bar(
    source_rating_df["Evaluator Source"],
    source_rating_df["Rating"]
)
ax3.set_ylim(0, 5)
ax3.set_ylabel("Rating")
ax3.set_title("Multi-Source Behavior Ratings")
st.pyplot(fig3)


# ==============================
# Module 3: Bias Detection
# ==============================

st.header("Module 3: Bias Detection and Fairness")

bias_flag = int(bias_employee["bias_flag"])
bias_type = bias_employee["bias_type"]
fairness_score = bias_employee["fairness_score"]

if bias_flag == 1:
    bias_status = "Possible Bias Risk Detected"
else:
    bias_status = "No Major Bias Detected"

col1, col2, col3 = st.columns(3)

col1.metric("Bias Status", bias_status)
col2.metric("Fairness Score", round(fairness_score, 2))

if "predicted_bias_flag" in bias_df.columns:
    col3.metric("Predicted Bias Flag", int(bias_employee["predicted_bias_flag"]))
else:
    col3.metric("Predicted Bias Flag", "N/A")

st.subheader("Bias Type / Explanation")
st.write(bias_type)

st.subheader("Bias Detection Signals")

bias_signal_cols = [
    "manager_rating",
    "peer_rating",
    "subordinate_rating",
    "overall_rating_avg",
    "rating_variance",
    "manager_peer_gap",
    "manager_subordinate_gap",
    "peer_subordinate_gap",
    "sentiment_score",
    "text_bias_score",
    "gender_rating_gap",
    "ethnicity_rating_gap",
    "language_rating_gap",
    "age_group_rating_gap",
    "group_bias_score"
]

available_bias_cols = [
    col for col in bias_signal_cols
    if col in bias_df.columns
]

st.dataframe(
    bias_employee[available_bias_cols].to_frame(name="Value")
)

st.subheader("Rating Comparison for Bias Review")

bias_rating_df = pd.DataFrame({
    "Evaluation Source": ["Manager", "Peer", "Subordinate"],
    "Rating": [
        bias_employee["manager_rating"],
        bias_employee["peer_rating"],
        bias_employee["subordinate_rating"]
    ]
})

fig4, ax4 = plt.subplots(figsize=(6, 4))
ax4.bar(
    bias_rating_df["Evaluation Source"],
    bias_rating_df["Rating"]
)
ax4.set_ylim(0, 5)
ax4.set_ylabel("Rating")
ax4.set_title("Manager vs Peer vs Subordinate Ratings")
st.pyplot(fig4)

if bias_flag == 1:
    st.warning(
        "This appraisal record contains possible bias indicators. "
        "Human review is recommended before final appraisal decisions."
    )
else:
    st.success(
        "No major bias indicator was detected. The appraisal record appears acceptable for normal review."
    )


# ==============================
# Module 4: Decision Arbitration
# ==============================

st.header("Module 4: Decision Arbitration and Opinion Dynamics")

col1, col2, col3 = st.columns(3)

col1.metric("Conflict Flag", int(decision_employee["conflict_flag"]))
col2.metric("Conflict Score", round(decision_employee["conflict_score"], 2))
col3.metric("Power Distance Risk", decision_employee["power_distance_risk"])

col4, col5, col6 = st.columns(3)

col4.metric("Final Recommended Score", round(decision_employee["final_recommended_score"], 2))
col5.metric("Final Recommended Class", decision_employee["final_recommended_class"])
col6.metric("Arbitration Decision", decision_employee["arbitration_decision"])

st.subheader("Conflict Type")
st.write(decision_employee["conflict_type"])

st.subheader("Opinion Dynamics Pattern")
st.write(decision_employee["opinion_dynamics_pattern"])

st.subheader("Institution Culture Pattern")
st.write(decision_employee["culture_pattern"])

st.subheader("Opinion Dynamics and Power Distance Signals")

opinion_signal_cols = [
    "manager_dominance_score",
    "peer_convergence_score",
    "subordinate_suppression_score",
    "social_protection_score",
    "power_distance_risk_score"
]

available_opinion_cols = [
    col for col in opinion_signal_cols
    if col in decision_df.columns
]

st.dataframe(
    decision_employee[available_opinion_cols].to_frame(name="Value")
)

fig5, ax5 = plt.subplots(figsize=(8, 4))
ax5.bar(
    available_opinion_cols,
    [decision_employee[col] for col in available_opinion_cols]
)
ax5.set_ylim(0, 100)
ax5.set_ylabel("Score")
ax5.set_title("Opinion Dynamics / Power Distance Indicators")
plt.xticks(rotation=30, ha="right")
st.pyplot(fig5)

st.subheader("Final Decision Explanation")
st.write(decision_employee["decision_explanation"])

if decision_employee["arbitration_decision"] == "Request HR Review":
    st.error("Final recommendation: HR review required before final appraisal approval.")
elif decision_employee["arbitration_decision"] in ["Adjust Manager Weight", "Adjust Peer Weight"]:
    st.warning("Final recommendation: Adjust evaluator weighting before final appraisal approval.")
else:
    st.success("Final recommendation: Appraisal can proceed with the suggested decision.")


# ==============================
# Feedback text
# ==============================

st.header("Feedback Text Used for Analysis")

if "feedback_text_combined" in bias_df.columns:
    st.write(bias_employee["feedback_text_combined"])
elif "feedback_text_combined" in behavior_df.columns:
    st.write(behavior_employee["feedback_text_combined"])
else:
    st.warning("Combined feedback text column not found.")


# ==============================
# Footer
# ==============================

st.markdown("---")
st.caption(
    "This demo is a conceptual AI-based appraisal decision-support system. "
    "It flags possible risks and recommendations for human review; it does not make final HR decisions automatically."
)