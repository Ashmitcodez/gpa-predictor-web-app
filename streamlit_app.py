import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor  # XGBoost model

# Set font to Arial for consistency and readability
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load and preprocess main GPA cutoff dataset
raw = pd.read_csv("uoa_engineering_gpa_data.csv")

# Load grade distribution data
grades = pd.read_excel("grade distribution - 2018 to 2024.xlsx", sheet_name="Sheet1")
grades = grades.dropna(subset=["Year"])
grades["Year"] = grades["Year"].astype(int)
grades["CohortYear"] = grades["Year"] + 1

# Define grade points for weighted GPA
grade_points = {
    "A+": 9, "A": 8, "A-": 7,
    "B+": 6, "B": 5, "B-": 4,
    "C+": 3, "C": 2, "C-": 1,
    "D+": 0, "D": 0, "D-": 0
}

# Ensure all grade columns exist
missing_cols = [g for g in grade_points if g not in grades.columns]
if missing_cols:
    st.error(f"Missing grade columns in data: {missing_cols}. Cannot proceed.")
    st.stop()

# Fill missing grade values with zeros
for g in grade_points:
    grades[g] = grades[g].fillna(0)

# Calculate total grades and weighted sum
grades["TotalGrades"] = sum(grades[g] for g in grade_points)
grades["WeightedSum"] = sum(grades[g] * p for g, p in grade_points.items())

# Remove invalid rows with zero total grades
if (grades["TotalGrades"] <= 0).any():
    st.warning("Some rows in grade data had zero total grades and were removed.")
grades = grades[grades["TotalGrades"] > 0]

# Compute additional features for each course row
grades["AvgCourseGPA_row"] = grades["WeightedSum"] / grades["TotalGrades"]
grades["PassRate_row"] = grades["Successful Completions"] / grades["Course Headcount"]
grades["PctA_row"] = (grades["A+"] + grades["A"] + grades["A-"]) / grades["TotalGrades"]
grades["PctB_row"] = (grades["B+"] + grades["B"] + grades["B-"]) / grades["TotalGrades"]
grades["PctCOrLower_row"] = (grades["C+"] + grades["C"] + grades["C-"] +
                             grades["D+"] + grades["D"] + grades["D-"]) / grades["TotalGrades"]

# Compute median GPA per course row
def row_median(row):
    values = []
    for g, p in grade_points.items():
        values.extend([p] * int(row[g]))
    if not values:
        return np.nan
    return float(np.median(values))

grades["MedianCourseGPA_row"] = grades.apply(row_median, axis=1)

# Aggregate by cohort year
var_by_year = grades.groupby("CohortYear")["AvgCourseGPA_row"].var().reset_index()
var_by_year.rename(columns={"AvgCourseGPA_row": "VarGPA"}, inplace=True)
agg = grades.groupby("CohortYear").agg(
    AvgCourseGPA=("AvgCourseGPA_row", "mean"),
    MedianCourseGPA=("MedianCourseGPA_row", "mean"),
    PassRate=("PassRate_row", "mean"),
    PctA=("PctA_row", "mean"),
    PctB=("PctB_row", "mean"),
    PctCOrLower=("PctCOrLower_row", "mean")
).reset_index()
agg = agg.merge(var_by_year, on="CohortYear", how="left")

# Merge features into main dataset
raw = raw.merge(agg, left_on="Year", right_on="CohortYear", how="left").drop(columns=["CohortYear"])

# Prepare data for modeling
data = raw.dropna(subset=["LowerBoundGPA"])
data = pd.get_dummies(data, columns=["Specialisation"], drop_first=True)

y = data["LowerBoundGPA"]
X = data.drop(columns=["LowerBoundGPA", "Year"])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.1)
}
for name, model in models.items():
    model.fit(X_train, y_train)

# App introduction
st.title("UoA Engineering GPA Predictor")
st.markdown("""
This tool predicts minimum GPA cutoffs for engineering specialisations at the University of Auckland.

It has two main modes:
1. Explore historical data (2019–2025): adjust factors and see what GPA cutoff would have been predicted, with metrics and plots showing model accuracy.
2. Predict future requirements (for example, 2026): enter your assumptions for seats, cohort size, popularity, and grade-distribution metrics to get a predicted cutoff.

Features used in the model:
- SeatsAvailable, CohortSize, PopularityScore
- AvgCourseGPA (average GPA across courses)
- MedianCourseGPA (median GPA across courses)
- PassRate (proportion of passes)
- PctA (percent of A-range grades)
- PctB (percent of B-range grades)
- PctCOrLower (percent of C or lower)
- VarGPA (variance in GPA across courses)

Model choices include Decision Tree, Random Forest, Linear Regression, Gradient Boosting, and XGBoost.
""")

# Historical data exploration
st.header("Explore Historical Data")
st.markdown("""
**How historical prediction works:**  
The model is trained using past records (2019–2025) where the GPA cutoff is already known.  
Each record corresponds to a particular specialisation and year, but the model does **not use the year itself** as a feature.  
Instead, it uses the conditions for that record — like seats available, cohort size, popularity, and grade statistics — to learn general patterns.

When you explore historical data here, you’re not asking the model to predict for a specific year.  
Instead, you’re seeing **how well the trained model can estimate known past cutoffs given those conditions**, and you can experiment by adjusting those conditions to see how the prediction would change.
""")

col1, col2, col3 = st.columns(3)
with col1:
    seats = st.slider("Seats Available", 30, 250, 100, key="hist_seats")
with col2:
    cohort = st.slider("Cohort Size", 800, 1200, 1000, key="hist_cohort")
with col3:
    popularity = st.slider("Popularity Score", 1.0, 10.0, 5.0, step=0.5, key="hist_popularity")

spec = st.selectbox(
    "Select Specialisation",
    sorted([c for c in data.columns if c.startswith("Specialisation_")]),
    key="hist_spec"
)

input_dict = {
    "SeatsAvailable": seats,
    "CohortSize": cohort,
    "PopularityScore": popularity,
    "AvgCourseGPA": data["AvgCourseGPA"].mean(),
    "MedianCourseGPA": data["MedianCourseGPA"].mean(),
    "PassRate": data["PassRate"].mean(),
    "PctA": data["PctA"].mean(),
    "PctB": data["PctB"].mean(),
    "PctCOrLower": data["PctCOrLower"].mean(),
    "VarGPA": data["VarGPA"].mean()
}
for col in data.columns:
    if col.startswith("Specialisation_"):
        input_dict[col] = 1 if col == spec else 0
input_df = pd.DataFrame([input_dict])

model_choice = st.selectbox("Select a model for prediction:", list(models.keys()), key="hist_model")
model = models[model_choice]

st.subheader("Predicted GPA cutoff (based on inputs)")
pred = model.predict(input_df)[0]
st.metric(label=f"{model_choice}", value=f"{pred:.2f}")

y_pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)
st.subheader("Model Performance on Past Data")
st.write(f"MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")

with st.expander("How to interpret these metrics"):
    st.markdown("""
MAE: On average, how far off predictions are in GPA points. Example: MAE = 0.3 means off by about 0.3 GPA points on average.  
RMSE: Similar to MAE but larger errors have more weight. Example: RMSE = 0.5 means typical error is about 0.5 GPA points.  
R²: Shows how much variation in GPA cutoffs is explained by the model (closer to 1 is better).
""")

st.subheader("Actual vs Predicted (historical data)")
fig1, ax1 = plt.subplots(figsize=(6,6))
ax1.scatter(y_test, y_pred_test, alpha=0.6)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
ax1.set_xlabel("Actual GPA")
ax1.set_ylabel("Predicted GPA")
ax1.set_title(f"Actual vs Predicted ({model_choice})")
st.pyplot(fig1)

with st.expander("How to interpret this plot"):
    st.markdown("""
Each point represents a past case. Points near the red dashed line mean the model predicted accurately. Points further away mean larger prediction errors.
""")

if hasattr(model, "feature_importances_"):
    st.subheader("Which factors matter most?")
    feat_df = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
    feat_df = feat_df.sort_values(by="Importance", ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.barplot(x="Importance", y="Feature", data=feat_df, ax=ax2)
    ax2.set_title(f"Top 10 Feature Importances ({model_choice})")
    st.pyplot(fig2)

    with st.expander("How to interpret this bar chart"):
        st.markdown("""
Each bar shows how important a feature was in making predictions. Longer bars mean the feature had more influence.
""")

# Future prediction section
st.header("Predict 2026 Requirement")
st.markdown("Enter your assumptions for 2026. Grade-related inputs are given as percentages for clarity.")

spec_2026 = st.selectbox(
    "Select Specialisation for 2026 Prediction",
    sorted([c for c in data.columns if c.startswith("Specialisation_")]),
    key="future_spec"
)

seats_2026 = st.number_input("Seats Available in 2026", min_value=30, max_value=250, value=100, key="future_seats")
cohort_2026 = st.number_input("Cohort Size in 2026", min_value=800, max_value=1500, value=1000, key="future_cohort")
popularity_2026 = st.number_input("Popularity Score in 2026", min_value=1.0, max_value=10.0, value=5.0, step=0.1, key="future_popularity")
avg_gpa_2026 = st.slider("Assumed AvgCourseGPA in 2026", 0.0, 9.0, float(data["AvgCourseGPA"].mean()), step=0.1, key="future_avg_gpa")
median_gpa_2026 = st.slider("Assumed MedianCourseGPA in 2026", 0.0, 9.0, float(data["MedianCourseGPA"].mean()), step=0.1, key="future_median_gpa")
pass_rate_2026 = st.number_input("Assumed PassRate in 2026 (%)", 0, 100, int(data["PassRate"].mean() * 100), step=1, key="future_passrate")
pct_a_2026 = st.number_input("Assumed % of A-range grades in 2026", 0, 100, int(data["PctA"].mean() * 100), step=1, key="future_pcta")
pct_b_2026 = st.number_input("Assumed % of B-range grades in 2026", 0, 100, int(data["PctB"].mean() * 100), step=1, key="future_pctb")
pct_c_2026 = st.number_input("Assumed % of C or lower grades in 2026", 0, 100, int(data["PctCOrLower"].mean() * 100), step=1, key="future_pctc")
var_gpa_2026 = st.slider("Assumed VarGPA in 2026", 0.0, 5.0, float(data["VarGPA"].mean()), step=0.1, key="future_vargpa")

model_choice_2026 = st.selectbox("Choose model for 2026 prediction:", list(models.keys()), key="future_model")
model_2026 = models[model_choice_2026]

if st.button("Predict 2026 GPA", key="predict_future"):
    # Validate percentage logic
    sum_pct = pct_a_2026 + pct_b_2026 + pct_c_2026
    fail_rate = 100 - pass_rate_2026
    error_msgs = []
    if abs(sum_pct - 100) > 2:
        error_msgs.append(f"The grade percentages (A+B+C or lower) add up to {sum_pct:.1f}%, expected 100%.")
    if pct_c_2026 > fail_rate + 1:
        error_msgs.append(f"C or lower grades ({pct_c_2026:.1f}%) exceed the failing percentage ({fail_rate:.1f}%) from PassRate.")
    if error_msgs:
        for m in error_msgs:
            st.error(m)
        st.stop()

    input_dict_2026 = {
        "SeatsAvailable": seats_2026,
        "CohortSize": cohort_2026,
        "PopularityScore": popularity_2026,
        "AvgCourseGPA": avg_gpa_2026,
        "MedianCourseGPA": median_gpa_2026,
        "PassRate": pass_rate_2026 / 100.0,
        "PctA": pct_a_2026 / 100.0,
        "PctB": pct_b_2026 / 100.0,
        "PctCOrLower": pct_c_2026 / 100.0,
        "VarGPA": var_gpa_2026
    }
    for col in data.columns:
        if col.startswith("Specialisation_"):
            input_dict_2026[col] = 1 if col == spec_2026 else 0
    input_df_2026 = pd.DataFrame([input_dict_2026])

    predicted_gpa_2026 = model_2026.predict(input_df_2026)[0]
    st.success(f"Predicted 2026 GPA cutoff for {spec_2026.replace('Specialisation_', '')}: {predicted_gpa_2026:.2f}")

    y_pred_test_2026 = model_2026.predict(X_test)
    mae_26 = mean_absolute_error(y_test, y_pred_test_2026)
    rmse_26 = np.sqrt(mean_squared_error(y_test, y_pred_test_2026))
    r2_26 = r2_score(y_test, y_pred_test_2026)
    st.markdown("Model performance on past data:")
    st.write(f"MAE: {mae_26:.3f} | RMSE: {rmse_26:.3f} | R²: {r2_26:.3f}")

    with st.expander("How to interpret these metrics"):
        st.markdown("""
Mean Absolute Error: average error in GPA points.  
Root Mean Squared Error: similar to MAE but gives more weight to larger errors.  
R²: proportion of variation explained by the model (closer to 1 means better).
""")

    if hasattr(model_2026, "feature_importances_"):
        st.subheader("Which factors mattered the most?")
        fi_df = pd.DataFrame({"Feature": X.columns, "Importance": model_2026.feature_importances_})
        fi_df = fi_df.sort_values(by="Importance", ascending=False).head(10)
        fig3, ax3 = plt.subplots(figsize=(8,6))
        sns.barplot(x="Importance", y="Feature", data=fi_df, ax=ax3)
        ax3.set_title(f"Top 10 Feature Importances ({model_choice_2026})")
        st.pyplot(fig3)

        with st.expander("How to interpret this bar chart"):
            st.markdown("""
Each bar shows how important that feature was in making predictions. Longer bars mean the feature was more important to the model.
""")
